 // ----------------------------------------------------------------------
//
// definition of ParameterSet's function members
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/Utilities/interface/Digest.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "FWCore/ParameterSet/interface/split.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"

#include <algorithm>
#include <iostream>

#include <sstream>

// ----------------------------------------------------------------------
// class invariant checker
// ----------------------------------------------------------------------

namespace edm {

  void
  ParameterSet::invalidateRegistration(std::string const& nameOfTracked) const {
    // We have added a new parameter.  Invalidate the ID.
    if(isRegistered()) {
      trackedID();
      id_ = ParameterSetID();
    }
    if (!nameOfTracked.empty() && trackedID_.isValid()) {
      // We have added a new tracked parameter.  Invalidate the tracked ID.
      trackedID_ = ParameterSetID();
      // Give a warning (informational for now)
      LogInfo("ParameterSet")    << "Warning: You have added a new tracked parameter\n"
				 <<  "'" << nameOfTracked << "' to a previously registered parameter set.\n"
				 << "This is a bad idea because the new parameter(s) will not be recorded.\n"
				 << "Use the forthcoming ParameterSetDescription facility instead.\n"
				 << "A warning is given only for the first such parameter in a pset.\n";
    }
    assert(!isRegistered());
  }

  // ----------------------------------------------------------------------
  // constructors
  // ----------------------------------------------------------------------

  ParameterSet::ParameterSet() :
    tbl_(),
    psetTable_(),
    vpsetTable_(),
    isFullyTracked_(True),
    id_(),
    trackedID_()
  {
  }

  // ----------------------------------------------------------------------
  // from coded string

  ParameterSet::ParameterSet(std::string const& code) :
    tbl_(),
    psetTable_(),
    vpsetTable_(),
    isFullyTracked_(True),
    id_(),
    trackedID_()
  {
    if(!fromString(code)) {
      throw edm::Exception(errors::Configuration,"InvalidInput")
	<< "The encoded configuration string "
	<< "passed to a ParameterSet during construction is invalid:\n"
	<< code;
    }
  }

  ParameterSet::~ParameterSet() {}

  ParameterSet const& ParameterSet::registerIt() {
    if(!isRegistered()) {
      calculateID();
      pset::Registry::instance()->insertMapped(*this);
    }
    return *this;
  }

  void ParameterSet::calculateID() {
    // make sure contained PSets are updated
    for(psettable::iterator i = psetTable_.begin(), e = psetTable_.end(); i != e; ++i) {
      if (!i->second.pset().isRegistered()) {
	i->second.pset().registerIt();
	i->second.updateID();
      }
    }

    for(vpsettable::iterator i = vpsetTable_.begin(), e = vpsetTable_.end(); i != e; ++i) {
      for (std::vector<ParameterSetEntry>::iterator
	    it = i->second.psetEntries().begin(), et = i->second.psetEntries().end();
	    it != et; ++it) {
	if (!it->pset().isRegistered()) {
          it->pset().registerIt();
          it->updateID();
	}
      }
    }

    std::string stringrep;
    toString(stringrep);
    cms::Digest md5alg(stringrep);
    id_ = ParameterSetID(md5alg.digest().toString());
    assert(isRegistered());
  }

  // ----------------------------------------------------------------------
  // identification
  ParameterSetID
  ParameterSet::id() const {
    // checks if valid
    if (!isRegistered()) {
      throw edm::Exception(edm::errors::LogicError)
        << "ParameterSet::id() called prematurely\n"
        << "before ParameterSet::registerIt() has been called.\n";
    }
    return id_;
  }

  ParameterSetID
  ParameterSet::trackedID() const {
    if (!trackedID_.isValid()) {
      if (!isRegistered()) {
	throw edm::Exception(edm::errors::LogicError)
          << "ParameterSet::trackedID() called prematurely\n"
          << "before ParameterSet::registerIt() has been called.\n";
      }
      if (isFullyTracked_ == True) {
        trackedID_ = id_;
      } else {
	ParameterSet pset = trackedPart();
	pset.registerIt();
        trackedID_ = pset.id_;
        isFullyTracked_ = (trackedID_ == id_ ? True : False);
      }
    }
    return trackedID_;
  }

  void ParameterSet::setID(ParameterSetID const& id) const {
    id_ = id;
  }

  bool
  ParameterSet::isFullyTracked() const {
    if (isFullyTracked_ == Unknown) {
      isFullyTracked_ = (trackedID() == id() ? True : False);
    }
    return (isFullyTracked_ == True);
  }

  // ----------------------------------------------------------------------
  // Entry-handling
  // ----------------------------------------------------------------------

  Entry const*
  ParameterSet::getEntryPointerOrThrow_(char const* name) const {
    return getEntryPointerOrThrow_(std::string(name));
  }

  Entry const*
  ParameterSet::getEntryPointerOrThrow_(std::string const& name) const {
    Entry const* result = retrieveUntracked(name);
    if (result == 0)
      throw edm::Exception(errors::Configuration, "MissingParameter:")
	<< "The required parameter '" << name
	<< "' was not specified.\n";
    return result;
  }

  template <class T, class U> T first(std::pair<T,U> const& p)
  { return p.first; }

  template <class T, class U> U second(std::pair<T,U> const& p)
  { return p.second; }

  Entry const&
  ParameterSet::retrieve(char const* name) const {
    return retrieve(std::string(name));
  }

  Entry const&
  ParameterSet::retrieve(std::string const& name) const {
    table::const_iterator  it = tbl_.find(name);
    if (it == tbl_.end()) {
	throw edm::Exception(errors::Configuration,"MissingParameter:")
	  << "Parameter '" << name
	  << "' not found.";
    }
    if (it->second.isTracked() == false) {
      if (name[0] == '@') {
	throw edm::Exception(errors::Configuration,"StatusMismatch:")
	  << "Framework Error:  Parameter '" << name
	  << "' is incorrectly designated as tracked in the framework.";
      } else {
	throw edm::Exception(errors::Configuration,"StatusMismatch:")
	  << "Parameter '" << name
	  << "' is designated as tracked in the code,\n"
          << "but is designated as untracked in the configuration file.\n"
          << "Please remove 'untracked' from the configuration file for parameter '"<< name << "'.";
      }
    }
    return it->second;
  }  // retrieve()

  Entry const* const
  ParameterSet::retrieveUntracked(char const* name) const {
    return retrieveUntracked(std::string(name));
  }

  Entry const* const
  ParameterSet::retrieveUntracked(std::string const& name) const {
    table::const_iterator  it = tbl_.find(name);

    if (it == tbl_.end()) return 0;
    if (it->second.isTracked()) {
      if (name[0] == '@') {
	throw edm::Exception(errors::Configuration,"StatusMismatch:")
	  << "Framework Error:  Parameter '" << name
	  << "' is incorrectly designated as untracked in the framework.";
      } else {
	throw edm::Exception(errors::Configuration,"StatusMismatch:")
	  << "Parameter '" << name
	  << "' is designated as untracked in the code,\n"
          << "but is not designated as untracked in the configuration file.\n"
          << "Please change the configuration file to 'untracked <type> " << name << "'.";
      }
    }
    return &it->second;
  }  // retrieve()

  ParameterSetEntry const&
  ParameterSet::retrieveParameterSet(std::string const& name) const {
    psettable::const_iterator it = psetTable_.find(name);
    if (it == psetTable_.end()) {
        throw edm::Exception(errors::Configuration,"MissingParameter:")
          << "ParameterSet '" << name
          << "' not found.";
    }
    if (it->second.isTracked() == false) {
      if (name[0] == '@') {
        throw edm::Exception(errors::Configuration,"StatusMismatch:")
          << "Framework Error:  ParameterSet '" << name
          << "' is incorrectly designated as tracked in the framework.";
      } else {
        throw edm::Exception(errors::Configuration,"StatusMismatch:")
          << "ParameterSet '" << name
          << "' is designated as tracked in the code,\n"
          << "but is designated as untracked in the configuration file.\n"
          << "Please remove 'untracked' from the configuration file for parameter '"<< name << "'.";
      }
    }
    return it->second;
  }  // retrieve()

  ParameterSetEntry const* const
  ParameterSet::retrieveUntrackedParameterSet(std::string const& name) const {
    psettable::const_iterator  it = psetTable_.find(name);

    if (it == psetTable_.end()) return 0;
    if (it->second.isTracked()) {
      if (name[0] == '@') {
        throw edm::Exception(errors::Configuration,"StatusMismatch:")
          << "Framework Error:  ParameterSet '" << name
          << "' is incorrectly designated as untracked in the framework.";
      } else {
        throw edm::Exception(errors::Configuration,"StatusMismatch:")
          << "ParameterSet '" << name
          << "' is designated as untracked in the code,\n"
          << "but is not designated as untracked in the configuration file.\n"
          << "Please change the configuration file to 'untracked <type> " << name << "'.";
      }
    }
    return &it->second;
  }  // retrieve()

  VParameterSetEntry const&
  ParameterSet::retrieveVParameterSet(std::string const& name) const {
    vpsettable::const_iterator it = vpsetTable_.find(name);
    if (it == vpsetTable_.end()) {
        throw edm::Exception(errors::Configuration,"MissingParameter:")
          << "VParameterSet '" << name
          << "' not found.";
    }
    if (it->second.isTracked() == false) {
      throw edm::Exception(errors::Configuration,"StatusMismatch:")
        << "VParameterSet '" << name
        << "' is designated as tracked in the code,\n"
        << "but is designated as untracked in the configuration file.\n"
        << "Please remove 'untracked' from the configuration file for parameter '"<< name << "'.";
    }
    return it->second;
  }  // retrieve()

  VParameterSetEntry const* const
  ParameterSet::retrieveUntrackedVParameterSet(std::string const& name) const {
    vpsettable::const_iterator it = vpsetTable_.find(name);

    if (it == vpsetTable_.end()) return 0;
    if (it->second.isTracked()) {
      throw edm::Exception(errors::Configuration,"StatusMismatch:")
        << "VParameterSet '" << name
        << "' is designated as untracked in the code,\n"
        << "but is not designated as untracked in the configuration file.\n"
        << "Please change the configuration file to 'untracked <type> " << name << "'.";
    }
    return &it->second;
  }  // retrieve()

  Entry const* const
  ParameterSet::retrieveUnknown(char const* name) const {
    return retrieveUnknown(std::string(name));
  }

  Entry const* const
  ParameterSet::retrieveUnknown(std::string const& name) const {
    table::const_iterator it = tbl_.find(name);
    if (it == tbl_.end()) {
      return 0;
    }
    return &it->second;
  }

  ParameterSetEntry const* const
  ParameterSet::retrieveUnknownParameterSet(std::string const& name) const {
    psettable::const_iterator  it = psetTable_.find(name);
    if (it == psetTable_.end()) {
      return 0;
    }
    return &it->second;
  }

  VParameterSetEntry const* const
  ParameterSet::retrieveUnknownVParameterSet(std::string const& name) const {
    vpsettable::const_iterator  it = vpsetTable_.find(name);
    if (it == vpsetTable_.end()) {
      return 0;
    }
    return &it->second;
  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  void
  ParameterSet::insert(bool okay_to_replace, char const* name, Entry const& value) {
    insert(okay_to_replace, std::string(name), value);
  }

  void
  ParameterSet::insert(bool okay_to_replace, std::string const& name, Entry const& value) {
    // We should probably get rid of 'okay_to_replace', which will
    // simplify the logic in this function.
    table::iterator  it = tbl_.find(name);

    if(it == tbl_.end())  {
      if(!tbl_.insert(std::make_pair(name, value)).second)
        throw edm::Exception(errors::Configuration,"InsertFailure")
	  << "cannot insert " << name
	  << " into a ParameterSet\n";
    }
    else if(okay_to_replace)  {
      it->second = value;
    }
  }  // insert()

  void ParameterSet::insertParameterSet(bool okay_to_replace, std::string const& name, ParameterSetEntry const& entry) {
    // We should probably get rid of 'okay_to_replace', which will
    // simplify the logic in this function.
    psettable::iterator it = psetTable_.find(name);

    if(it == psetTable_.end()) {
      if(!psetTable_.insert(std::make_pair(name, entry)).second)
        throw edm::Exception(errors::Configuration,"InsertFailure")
          << "cannot insert " << name
          << " into a ParameterSet\n";
    } else if(okay_to_replace) {
      it->second = entry;
    }
  }  // insert()

  void ParameterSet::insertVParameterSet(bool okay_to_replace, std::string const& name, VParameterSetEntry const& entry) {
    // We should probably get rid of 'okay_to_replace', which will
    // simplify the logic in this function.
    vpsettable::iterator it = vpsetTable_.find(name);

    if(it == vpsetTable_.end()) {
      if(!vpsetTable_.insert(std::make_pair(name, entry)).second)
        throw edm::Exception(errors::Configuration,"InsertFailure")
          << "cannot insert " << name
          << " into a VParameterSet\n";
    } else if(okay_to_replace) {
      it->second = entry;
    }
  }  // insert()

  void
  ParameterSet::augment(ParameterSet const& from) {
    // This preemptive invalidation may be more agressive than necessary.
    invalidateRegistration();

    if(&from == this) {
      return;
    }

    for(table::const_iterator b = from.tbl_.begin(), e = from.tbl_.end(); b != e; ++b) {
      this->insert(false, b->first, b->second);
    }
    for(psettable::const_iterator b = from.psetTable_.begin(), e = from.psetTable_.end(); b != e; ++b) {
      this->insertParameterSet(false, b->first, b->second);
    }
    for(vpsettable::const_iterator b = from.vpsetTable_.begin(), e = from.vpsetTable_.end(); b != e; ++b) {
      this->insertVParameterSet(false, b->first, b->second);
    }
  }  // augment()

  // ----------------------------------------------------------------------
  // coding
  // ----------------------------------------------------------------------

  void
  ParameterSet::toString(std::string& rep) const {
    // make sure the PSets get filled
    if (empty()) {
      rep += "<>";
      return;
    }
    size_t size = 1;
    for(table::const_iterator b = tbl_.begin(), e = tbl_.end(); b != e; ++b) {
      size += 2;
      size += b->first.size();
      size += b->second.sizeOfString();
    }
    for(psettable::const_iterator b = psetTable_.begin(), e = psetTable_.end(); b != e; ++b) {
      size += 2;
      size += b->first.size();
      size += b->second.sizeOfString();
    }
    for(vpsettable::const_iterator b = vpsetTable_.begin(), e = vpsetTable_.end(); b != e; ++b) {
      size += 2;
      size += b->first.size();
      size += b->second.sizeOfString();
    }

    rep.reserve(rep.size()+size);
    rep += '<';
    std::string start;
    std::string const between(";");
    for(table::const_iterator b = tbl_.begin(), e = tbl_.end(); b != e; ++b) {
      rep += start;
      rep += b->first;
      rep += '=';
      b->second.toString(rep);
      start = between;
    }
    for(psettable::const_iterator b = psetTable_.begin(), e = psetTable_.end(); b != e; ++b) {
      rep += start;
      rep += b->first;
      rep += '=';
      b->second.toString(rep);
      start = between;
    }
    for(vpsettable::const_iterator b = vpsetTable_.begin(), e = vpsetTable_.end(); b != e; ++b) {
      rep += start;
      rep += b->first;
      rep += '=';
      b->second.toString(rep);
      start = between;
    }

    rep += '>';
  }  // to_string()

  std::string
  ParameterSet::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  // ----------------------------------------------------------------------

  bool
  ParameterSet::fromString(std::string const& from) {
    std::vector<std::string> temp;
    if(!split(std::back_inserter(temp), from, '<', ';', '>'))
      return false;

    tbl_.clear();  // precaution
    for(std::vector<std::string>::const_iterator b = temp.begin(), e = temp.end(); b != e; ++b) {
      // locate required name/value separator
      std::string::const_iterator q = find_in_all(*b, '=');
      if(q == b->end())
        return false;

      // form name unique to this ParameterSet
      std::string  name = std::string(b->begin(), q);
      if(tbl_.find(name) != tbl_.end())
        return false;

      std::string rep(q+1, b->end());
      // entries are generically of the form tracked-type-rep
      if (rep[0] == '-') {
	isFullyTracked_ = False;
      }
      if(rep[1] == 'Q') {
        ParameterSetEntry psetEntry(rep);
        if(!psetTable_.insert(std::make_pair(name, psetEntry)).second) {
          return false;
	}
	isFullyTracked_ = isFullyTracked_ && Unknown;
      } else if(rep[1] == 'q') {
        VParameterSetEntry vpsetEntry(rep);
        if(!vpsetTable_.insert(std::make_pair(name, vpsetEntry)).second) {
          return false;
	}
	isFullyTracked_ = isFullyTracked_ && Unknown;
      } else if(rep[1] == 'P') {
        //old representation of ParameterSet, included for backwards-compatibility
        Entry value(name, rep);
        ParameterSetEntry psetEntry(value.getPSet(), value.isTracked());
        if(!psetTable_.insert(std::make_pair(name, psetEntry)).second) {
          return false;
        }
	isFullyTracked_ = isFullyTracked_ && Unknown;
      } else if(rep[1] == 'p') {
        //old representation of VParameterSet, included for backwards-compatibility
        Entry value(name, rep);
        VParameterSetEntry vpsetEntry(value.getVPSet(), value.isTracked());
        if(!vpsetTable_.insert(std::make_pair(name, vpsetEntry)).second) {
          return false;
        }
	isFullyTracked_ = isFullyTracked_ && Unknown;
      } else {
        // form value and insert name/value pair
        Entry  value(name, rep);
        if(!tbl_.insert(std::make_pair(name, value)).second) {
          return false;
        }
      }
    }

    return true;
  }  // from_string()

  std::vector<FileInPath>::size_type
  ParameterSet::getAllFileInPaths(std::vector<FileInPath>& output) const {
    std::vector<FileInPath>::size_type count = 0;
    table::const_iterator it = tbl_.begin();
    table::const_iterator end = tbl_.end();
    while (it != end) {
	Entry const& e = it->second;
	if (e.typeCode() == 'F') {
	    ++count;
	    output.push_back(e.getFileInPath());
	}
	++it;
    }
    return count;
  }

  std::vector<std::string>
  ParameterSet::getParameterNames() const {
    std::vector<std::string> returnValue;
    std::transform(tbl_.begin(), tbl_.end(), back_inserter(returnValue),
		   boost::bind(&std::pair<std::string const, Entry>::first,_1));
    std::transform(psetTable_.begin(), psetTable_.end(), back_inserter(returnValue),
                   boost::bind(&std::pair<std::string const, ParameterSetEntry>::first,_1));
    std::transform(vpsetTable_.begin(), vpsetTable_.end(), back_inserter(returnValue),
                   boost::bind(&std::pair<std::string const, VParameterSetEntry>::first,_1));
    return returnValue;
  }

  bool ParameterSet::exists(std::string const& parameterName) const {
    return(tbl_.find(parameterName) != tbl_.end() ||
     psetTable_.find(parameterName) != psetTable_.end() ||
    vpsetTable_.find(parameterName) != vpsetTable_.end());
  }

  ParameterSet
  ParameterSet::trackedPart() const {
    if (isFullyTracked_ == True) {
      return *this;
    }
    ParameterSet result;
    for(table::const_iterator tblItr = tbl_.begin(); tblItr != tbl_.end(); ++tblItr) {
      if(tblItr->second.isTracked()) {
        result.tbl_.insert(*tblItr);
      } else {
        isFullyTracked_ = False;
      }
    }
    for(psettable::const_iterator psetItr = psetTable_.begin(); psetItr != psetTable_.end(); ++psetItr) {
      if(psetItr->second.isTracked()) {
        result.addParameter<ParameterSet>(psetItr->first, psetItr->second.pset().trackedPart());
      } else {
        isFullyTracked_ = False;
      }
    }
    for(vpsettable::const_iterator vpsetItr = vpsetTable_.begin(); vpsetItr != vpsetTable_.end(); ++vpsetItr) {
      if(vpsetItr->second.isTracked()) {
	VParameterSet vresult;
	typedef std::vector<ParameterSetEntry> VPSE;
	typedef VPSE::const_iterator Iter;
	VPSE const& vpse = vpsetItr->second.psetEntries();
	for (Iter i = vpse.begin(), e = vpse.end(); i != e; ++i) {
	  vresult.push_back(i->pset().trackedPart());
	}
        result.addParameter<VParameterSet>(vpsetItr->first, vresult);
      } else {
        isFullyTracked_ = False;
      }
    }
    result.isFullyTracked_ = True;
    return result;
  }

  size_t
  ParameterSet::getParameterSetNames(std::vector<std::string>& output) {
    std::transform(psetTable_.begin(), psetTable_.end(), back_inserter(output),
                   boost::bind(&std::pair<std::string const, ParameterSetEntry>::first,_1));
    return output.size();
  }

  size_t
  ParameterSet::getParameterSetNames(std::vector<std::string>& output,
                                     bool trackiness) const {
    for(psettable::const_iterator psetItr = psetTable_.begin();
        psetItr != psetTable_.end(); ++psetItr) {
      if(psetItr->second.isTracked() == trackiness) {
        output.push_back(psetItr->first);
      }
    }
    return output.size();
  }

  size_t
  ParameterSet::getParameterSetVectorNames(std::vector<std::string>& output,
					    bool trackiness) const {
    for(vpsettable::const_iterator vpsetItr = vpsetTable_.begin();
         vpsetItr != vpsetTable_.end(); ++vpsetItr) {
      if(vpsetItr->second.isTracked() == trackiness) {
        output.push_back(vpsetItr->first);
      }
    }
    return output.size();
  }

  size_t
  ParameterSet::getNamesByCode_(char code,
				bool trackiness,
				std::vector<std::string>& output) const {
    size_t count = 0;
    if (code == 'Q') {
      return getParameterSetNames(output, trackiness);
    }
    if (code == 'q') {
      return getParameterSetVectorNames(output, trackiness);
    }
    table::const_iterator it = tbl_.begin();
    table::const_iterator end = tbl_.end();
    while (it != end) {
      Entry const& e = it->second;
      if (e.typeCode() == code &&
	  e.isTracked() == trackiness) { // if it is a vector of ParameterSet
	  ++count;
	  output.push_back(it->first); // save the name
      }
      ++it;
    }
    return count;
  }


  template <>
  std::vector<std::string> ParameterSet::getParameterNamesForType<FileInPath>(bool trackiness) const
  {
    std::vector<std::string> result;
    getNamesByCode_('F', trackiness, result);
    return result;
  }


  bool operator==(ParameterSet const& a, ParameterSet const& b) {
    // Maybe can replace this with comparison of id_ values.
    std::string aString;
    std::string bString;
    a.toString(aString);
    b.toString(bString);
    return aString == bString;
  }

  std::string ParameterSet::dump() const {
    std::ostringstream os;
    os << "{" << std::endl;
    for(table::const_iterator i = tbl_.begin(), e = tbl_.end(); i != e; ++i) {
      // indent a bit
      os << "  " << i->first << ": " << i->second << std::endl;
    }
    os << "}";
    os << "{" << std::endl;
    for(psettable::const_iterator i = psetTable_.begin(), e = psetTable_.end(); i != e; ++i) {
      // indent a bit
      std::string n = i->first;
      ParameterSetEntry const& pe = i->second;
      os << "  " << n << ": " << pe <<  std::endl;
    }
    os << "}";
    os << "{" << std::endl;
    for(vpsettable::const_iterator i = vpsetTable_.begin(), e = vpsetTable_.end(); i != e; ++i) {
      // indent a bit
      std::string n = i->first;
      VParameterSetEntry const& pe = i->second;
      os << "  " << n << ": " << pe <<  std::endl;
    }
    os << "}";
    return os.str();
  }

  std::ostream & operator<<(std::ostream & os, ParameterSet const& pset) {
    os << pset.dump();
    return os;
  }

  // Free function to return a parameterSet given its ID.
  ParameterSet
  getParameterSet(ParameterSetID const& id) {
    ParameterSet result;
    if(!pset::Registry::instance()->getMapped(id, result)) {
      throw edm::Exception(errors::Configuration,"MissingParameterSet:")
        << "Parameter Set ID '" << id << "' not found.";
    }
    return result;
  }

  void ParameterSet::deprecatedInputTagWarning(std::string const& name, 
					       std::string const& label) const {
    LogWarning("Configuration") << "Warning:\n\tstring " << name 
				<< " = \"" << label 
				<< "\"\nis deprecated, "
				<< "please update your config file to use\n\tInputTag " 
				<< name << " = " << label;
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  bool
  ParameterSet::getParameter<bool>(std::string const& name) const {
    return retrieve(name).getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  int
  ParameterSet::getParameter<int>(std::string const& name) const {
    return retrieve(name).getInt32();
  }

  template<>
  std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(std::string const& name) const {
    return retrieve(name).getVInt32();
  }
  
 // ----------------------------------------------------------------------
  // Int64, vInt64

  template<>
  boost::int64_t
  ParameterSet::getParameter<boost::int64_t>(std::string const& name) const {
    return retrieve(name).getInt64();
  }

  template<>
  std::vector<boost::int64_t>
  ParameterSet::getParameter<std::vector<boost::int64_t> >(std::string const& name) const {
    return retrieve(name).getVInt64();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  unsigned int
  ParameterSet::getParameter<unsigned int>(std::string const& name) const {
    return retrieve(name).getUInt32();
  }
  
  template<>
  std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(std::string const& name) const {
    return retrieve(name).getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  boost::uint64_t
  ParameterSet::getParameter<boost::uint64_t>(std::string const& name) const {
    return retrieve(name).getUInt64();
  }

  template<>
  std::vector<boost::uint64_t>
  ParameterSet::getParameter<std::vector<boost::uint64_t> >(std::string const& name) const {
    return retrieve(name).getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  double
  ParameterSet::getParameter<double>(std::string const& name) const {
    return retrieve(name).getDouble();
  }
  
  template<>
  std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(std::string const& name) const {
    return retrieve(name).getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  std::string
  ParameterSet::getParameter<std::string>(std::string const& name) const {
    return retrieve(name).getString();
  }
  
  template<>
  std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(std::string const& name) const {
    return retrieve(name).getVString();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  template <>
  FileInPath
  ParameterSet::getParameter<FileInPath>(std::string const& name) const {
    return retrieve(name).getFileInPath();
  }
  
  // ----------------------------------------------------------------------
  // InputTag

  template <>
  InputTag
  ParameterSet::getParameter<InputTag>(std::string const& name) const {
    Entry const& e_input = retrieve(name);
    switch (e_input.typeCode()) {
      case 't':   // InputTag
        return e_input.getInputTag();
      case 'S':   // string
        std::string const& label = e_input.getString();
	deprecatedInputTagWarning(name, label);
        return InputTag(label);
    }
    throw edm::Exception(errors::Configuration, "ValueError") << "type of " 
       << name << " is expected to be InputTag or string (deprecated)";

  }

  // ----------------------------------------------------------------------
  // VInputTag

  template <>
  std::vector<InputTag>
  ParameterSet::getParameter<std::vector<InputTag> >(std::string const& name) const {
    return retrieve(name).getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID

  template <>
  EventID
  ParameterSet::getParameter<EventID>(std::string const& name) const {
    return retrieve(name).getEventID();
  }

  // ----------------------------------------------------------------------
  // VEventID

  template <>
  std::vector<EventID>
  ParameterSet::getParameter<std::vector<EventID> >(std::string const& name) const {
    return retrieve(name).getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template <>
  LuminosityBlockID
  ParameterSet::getParameter<LuminosityBlockID>(std::string const& name) const {
    return retrieve(name).getLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template <>
  std::vector<LuminosityBlockID>
  ParameterSet::getParameter<std::vector<LuminosityBlockID> >(std::string const& name) const {
    return retrieve(name).getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  ParameterSet
  ParameterSet::getParameter<ParameterSet>(std::string const& name) const {
    return getParameterSet(name);
  }
  
  template<>
  VParameterSet
  ParameterSet::getParameter<VParameterSet>(std::string const& name) const {
    return getParameterSetVector(name);
  }
  
  // untracked parameters
  
  // ----------------------------------------------------------------------
  // Bool, vBool

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name, bool const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getBool();
  }

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getBool();
  }
  
  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  int
  ParameterSet::getUntrackedParameter<int>(std::string const& name, int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt32();
  }

  template<>
  int
  ParameterSet::getUntrackedParameter<int>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInt32();
  }

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name, std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt32();
  }

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name, unsigned int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt32();
  }

  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getUInt32();
  }
  
  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt32();
  }

  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVUInt32();
  }

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(std::string const& name, boost::uint64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt64();
  }

  template<>
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getUInt64();
  }

  template<>
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(std::string const& name, std::vector<boost::uint64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt64();
  }

  template<>
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Int64, Vint64

  template<>
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(std::string const& name, boost::int64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt64();
  }

  template<>
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInt64();
  }

  template<>
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(std::string const& name, std::vector<boost::int64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt64();
  }

  template<>
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  double
  ParameterSet::getUntrackedParameter<double>(std::string const& name, double const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getDouble();
  }

  template<>
  double
  ParameterSet::getUntrackedParameter<double>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getDouble();
  }  
  
  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name); return entryPtr == 0 ? defaultValue : entryPtr->getVDouble(); 
  }

  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name, std::string const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getString();
  }

  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getString();
  }
  
  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVString();
  }

  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVString();
  }

  // ----------------------------------------------------------------------
  //  FileInPath

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name, FileInPath const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getFileInPath();
  }

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(std::string const& name, InputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInputTag();
  }

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInputTag();
  }

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name, 
                                      std::vector<InputTag> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInputTag();
  }

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(std::string const& name, EventID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getEventID();
  }

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getEventID();
  }

  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name,
                                      std::vector<EventID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVEventID();
  }

  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name, LuminosityBlockID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getLuminosityBlockID();
  }

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockID();
  }

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(std::string const& name,
                                      std::vector<LuminosityBlockID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVLuminosityBlockID();
  }

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockID();
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  bool
  ParameterSet::getParameter<bool>(char const* name) const {
    return retrieve(name).getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  int
  ParameterSet::getParameter<int>(char const* name) const {
    return retrieve(name).getInt32();
  }

  template<>
  std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(char const* name) const {
    return retrieve(name).getVInt32();
  }
  
 // ----------------------------------------------------------------------
  // Int64, vInt64

  template<>
  boost::int64_t
  ParameterSet::getParameter<boost::int64_t>(char const* name) const {
    return retrieve(name).getInt64();
  }

  template<>
  std::vector<boost::int64_t>
  ParameterSet::getParameter<std::vector<boost::int64_t> >(char const* name) const {
    return retrieve(name).getVInt64();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  unsigned int
  ParameterSet::getParameter<unsigned int>(char const* name) const {
    return retrieve(name).getUInt32();
  }
  
  template<>
  std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(char const* name) const {
    return retrieve(name).getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  boost::uint64_t
  ParameterSet::getParameter<boost::uint64_t>(char const* name) const {
    return retrieve(name).getUInt64();
  }

  template<>
  std::vector<boost::uint64_t>
  ParameterSet::getParameter<std::vector<boost::uint64_t> >(char const* name) const {
    return retrieve(name).getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  double
  ParameterSet::getParameter<double>(char const* name) const {
    return retrieve(name).getDouble();
  }
  
  template<>
  std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(char const* name) const {
    return retrieve(name).getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  std::string
  ParameterSet::getParameter<std::string>(char const* name) const {
    return retrieve(name).getString();
  }
  
  template<>
  std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(char const* name) const {
    return retrieve(name).getVString();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  template <>
  FileInPath
  ParameterSet::getParameter<FileInPath>(char const* name) const {
    return retrieve(name).getFileInPath();
  }
  
  // ----------------------------------------------------------------------
  // InputTag

  template <>
  InputTag
  ParameterSet::getParameter<InputTag>(char const* name) const {
    Entry const& e_input = retrieve(name);
    switch (e_input.typeCode()) 
    {
      case 't':   // InputTag
        return e_input.getInputTag();
      case 'S':   // string
        std::string const& label = e_input.getString();
	deprecatedInputTagWarning(name, label);
        return InputTag(label);
    }
    throw edm::Exception(errors::Configuration, "ValueError") << "type of " 
       << name << " is expected to be InputTag or string (deprecated)";
  }

  // ----------------------------------------------------------------------
  // VInputTag

  template <>
  std::vector<InputTag>
  ParameterSet::getParameter<std::vector<InputTag> >(char const* name) const {
    return retrieve(name).getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID

  template <>
  EventID
  ParameterSet::getParameter<EventID>(char const* name) const {
    return retrieve(name).getEventID();
  }

  // ----------------------------------------------------------------------
  // VEventID

  template <>
  std::vector<EventID>
  ParameterSet::getParameter<std::vector<EventID> >(char const* name) const {
    return retrieve(name).getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template <>
  LuminosityBlockID
  ParameterSet::getParameter<LuminosityBlockID>(char const* name) const {
    return retrieve(name).getLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template <>
  std::vector<LuminosityBlockID>
  ParameterSet::getParameter<std::vector<LuminosityBlockID> >(char const* name) const {
    return retrieve(name).getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  ParameterSet
  ParameterSet::getParameter<ParameterSet>(char const* name) const {
    return getParameterSet(name);
  }
  
  template<>
  VParameterSet
  ParameterSet::getParameter<VParameterSet>(char const* name) const {
    return getParameterSetVector(name);
  }

  // untracked parameters
  
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(char const* name, bool const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getBool();
  }

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getBool();
  }
  
  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  int
  ParameterSet::getUntrackedParameter<int>(char const* name, int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt32();
  }

  template<>
  int
  ParameterSet::getUntrackedParameter<int>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInt32();
  }

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name, std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt32();
  }

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(char const* name, unsigned int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt32();
  }

  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getUInt32();
  }
  
  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt32();
  }

  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVUInt32();
  }

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(char const* name, boost::uint64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt64();
  }

  template<>
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getUInt64();
  }

  template<>
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(char const* name, std::vector<boost::uint64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt64();
  }

  template<>
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Int64, Vint64

  template<>
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(char const* name, boost::int64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt64();
  }

  template<>
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInt64();
  }

  template<>
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(char const* name, std::vector<boost::int64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt64();
  }

  template<>
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  double
  ParameterSet::getUntrackedParameter<double>(char const* name, double const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getDouble();
  }

  template<>
  double
  ParameterSet::getUntrackedParameter<double>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getDouble();
  }  
  
  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name); return entryPtr == 0 ? defaultValue : entryPtr->getVDouble(); 
  }

  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(char const* name, std::string const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getString();
  }

  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getString();
  }
  
  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVString();
  }

  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVString();
  }

  // ----------------------------------------------------------------------
  //  FileInPath

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(char const* name, FileInPath const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getFileInPath();
  }

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(char const* name, InputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInputTag();
  }

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInputTag();
  }

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name, 
                                      std::vector<InputTag> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInputTag();
  }

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(char const* name, EventID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getEventID();
  }

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getEventID();
  }

  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name,
                                      std::vector<EventID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVEventID();
  }

  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name, LuminosityBlockID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getLuminosityBlockID();
  }

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockID();
  }

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(char const* name,
                                      std::vector<LuminosityBlockID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVLuminosityBlockID();
  }

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockID();
  }
  
  // ----------------------------------------------------------------------
  // PSet, vPSet

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(char const* name, ParameterSet const& defaultValue) const {
    return getUntrackedParameterSet(name, defaultValue);
  }

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(char const* name, VParameterSet const& defaultValue) const {
    return getUntrackedParameterSetVector(name, defaultValue);
  }

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name, ParameterSet const& defaultValue) const {
    return getUntrackedParameterSet(name, defaultValue);
  }

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(std::string const& name, VParameterSet const& defaultValue) const {
    return getUntrackedParameterSetVector(name, defaultValue);
  }

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(char const* name) const {
    return getUntrackedParameterSet(name);
  }

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(char const* name) const {
    return getUntrackedParameterSetVector(name);
  }

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name) const {
    return getUntrackedParameterSet(name);
  }

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(std::string const& name) const {
    return getUntrackedParameterSetVector(name);
  }

//----------------------------------------------------------------------------------
// specializations for addParameter and addUntrackedParameter

  template <>
  void
  ParameterSet::addParameter<ParameterSet>(std::string const& name, ParameterSet value) {
    invalidateRegistration(name);
    insertParameterSet(true, name, ParameterSetEntry(value, true));
    isFullyTracked_ = isFullyTracked_ && value.isFullyTracked_;
  }

  template <>
  void
  ParameterSet::addParameter<VParameterSet>(std::string const& name, VParameterSet value) {
    invalidateRegistration(name);
    insertVParameterSet(true, name, VParameterSetEntry(value, true));
    isFullyTracked_ = isFullyTracked_ && Unknown;
  }

  template <>
  void
  ParameterSet::addParameter<ParameterSet>(char const* name, ParameterSet value) {
    invalidateRegistration(name);
    insertParameterSet(true, name, ParameterSetEntry(value, true));
    isFullyTracked_ = isFullyTracked_ && value.isFullyTracked_;
  }

  template <>
  void
  ParameterSet::addParameter<VParameterSet>(char const* name, VParameterSet value) {
    invalidateRegistration(name);
    insertVParameterSet(true, name, VParameterSetEntry(value, true));
    isFullyTracked_ = isFullyTracked_ && Unknown;
  }

  template <>
  void
  ParameterSet::addUntrackedParameter<ParameterSet>(std::string const& name, ParameterSet value) {
    invalidateRegistration();
    insertParameterSet(true, name, ParameterSetEntry(value, false));
    isFullyTracked_ = False;
  }

  template <>
  void
  ParameterSet::addUntrackedParameter<VParameterSet>(std::string const& name, VParameterSet value) {
    invalidateRegistration();
    insertVParameterSet(true, name, VParameterSetEntry(value, false));
    isFullyTracked_ = False;
  }

  template <>
  void
  ParameterSet::addUntrackedParameter<ParameterSet>(char const* name, ParameterSet value) {
    invalidateRegistration();
    insertParameterSet(true, name, ParameterSetEntry(value, false));
    isFullyTracked_ = False;
  }

  template <>
  void
  ParameterSet::addUntrackedParameter<VParameterSet>(char const* name, VParameterSet value) {
    invalidateRegistration();
    insertVParameterSet(true, name, VParameterSetEntry(value, false));
    isFullyTracked_ = False;
  }

//----------------------------------------------------------------------------------
// specializations for getParameterNamesForType

  template <>
  std::vector<std::string> 
  ParameterSet::getParameterNamesForType<ParameterSet>(bool trackiness) const {
    std::vector<std::string> output;
    getParameterSetNames(output, trackiness);
    return output; 
  }

  template <>
  std::vector<std::string> 
  ParameterSet::getParameterNamesForType<VParameterSet>(bool trackiness) const {
    std::vector<std::string> output;
    getParameterSetVectorNames(output, trackiness);
    return output; 
  }

  ParameterSet const&
  ParameterSet::getParameterSet(std::string const& name) const {
    return retrieveParameterSet(name).pset();
  }

  ParameterSet const&
  ParameterSet::getParameterSet(char const* name) const {
    return retrieveParameterSet(name).pset();
  }
  ParameterSet const&
  ParameterSet::getUntrackedParameterSet(std::string const& name, ParameterSet const& defaultValue) const {
    return getUntrackedParameterSet(name.c_str(), defaultValue);
  }

  ParameterSet const&
  ParameterSet::getUntrackedParameterSet(char const* name, ParameterSet const& defaultValue) const {
    ParameterSetEntry const* entryPtr = retrieveUntrackedParameterSet(name);
    return entryPtr == 0 ? defaultValue : entryPtr->pset();
  }

  ParameterSet const&
  ParameterSet::getUntrackedParameterSet(std::string const& name) const {
    return getUntrackedParameterSet(name.c_str());
  }

  ParameterSet const&
  ParameterSet::getUntrackedParameterSet(char const* name) const {
    ParameterSetEntry const* result = retrieveUntrackedParameterSet(name);
    if (result == 0)
      throw edm::Exception(errors::Configuration, "MissingParameter:")
        << "The required ParameterSet '" << name << "' was not specified.\n";
    return result->pset();
  }

  VParameterSet const&
  ParameterSet::getParameterSetVector(std::string const& name) const {
    return retrieveVParameterSet(name).vpset();
  }

  VParameterSet const&
  ParameterSet::getParameterSetVector(char const* name) const {
    return retrieveVParameterSet(name).vpset();
  }

  VParameterSet const&
  ParameterSet::getUntrackedParameterSetVector(std::string const& name, VParameterSet const& defaultValue) const {
    return getUntrackedParameterSetVector(name.c_str(), defaultValue);
  }

  VParameterSet const&
  ParameterSet::getUntrackedParameterSetVector(char const* name, VParameterSet const& defaultValue) const {
    VParameterSetEntry const* entryPtr = retrieveUntrackedVParameterSet(name);
    return entryPtr == 0 ? defaultValue : entryPtr->vpset();
  }

  VParameterSet const&
  ParameterSet::getUntrackedParameterSetVector(std::string const& name) const {
    return getUntrackedParameterSetVector(name.c_str());
  }

  VParameterSet const&
  ParameterSet::getUntrackedParameterSetVector(char const* name) const {
    VParameterSetEntry const* result = retrieveUntrackedVParameterSet(name);
    if (result == 0)
      throw edm::Exception(errors::Configuration, "MissingParameter:")
        << "The required ParameterSetVector '" << name << "' was not specified.\n";
    return result->vpset();
  }

//----------------------------------------------------------------------------------
  ParameterSet::Bool
  operator&&(ParameterSet::Bool a, ParameterSet::Bool b) {
    if (a == ParameterSet::False || b == ParameterSet::False) {
      return ParameterSet::False;
    } else if (a == ParameterSet::Unknown || b == ParameterSet::Unknown) {
      return ParameterSet::Unknown;
    }
    return ParameterSet::True;
  }


} // namespace edm
