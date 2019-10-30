// ----------------------------------------------------------------------
//
// definition of ParameterSet's function members
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/split.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <cassert>

// ----------------------------------------------------------------------
// class invariant checker
// ----------------------------------------------------------------------

namespace edm {

  void ParameterSet::invalidateRegistration(std::string const& nameOfTracked) {
    // We have added a new parameter.  Invalidate the ID.
    if (isRegistered()) {
      id_ = ParameterSetID();
      if (!nameOfTracked.empty()) {
        // Give a warning (informational for now).
        LogInfo("ParameterSet") << "Warning: You have added a new tracked parameter\n"
                                << "'" << nameOfTracked << "' to a previously registered parameter set.\n"
                                << "This is a bad idea because the new parameter(s) will not be recorded.\n"
                                << "Use the forthcoming ParameterSetDescription facility instead.\n"
                                << "A warning is given only for the first such parameter in a pset.\n";
      }
    }
    assert(!isRegistered());
  }

  // ----------------------------------------------------------------------
  // constructors
  // ----------------------------------------------------------------------

  ParameterSet::ParameterSet() : tbl_(), psetTable_(), vpsetTable_(), id_() {}

  // ----------------------------------------------------------------------
  // from coded string

  ParameterSet::ParameterSet(std::string const& code) : tbl_(), psetTable_(), vpsetTable_(), id_() {
    if (!fromString(code)) {
      throw Exception(errors::Configuration, "InvalidInput")
          << "The encoded configuration string "
          << "passed to a ParameterSet during construction is invalid:\n"
          << code;
    }
  }

  // ----------------------------------------------------------------------
  // from coded string and ID.

  ParameterSet::ParameterSet(std::string const& code, ParameterSetID const& id)
      : tbl_(), psetTable_(), vpsetTable_(), id_(id) {
    if (!fromString(code)) {
      throw Exception(errors::Configuration, "InvalidInput")
          << "The encoded configuration string "
          << "passed to a ParameterSet during construction is invalid:\n"
          << code;
    }
  }

  void ParameterSet::registerFromString(std::string const& rep) {
    // from coded string.  Will cause registration
    cms::Digest dg(rep);
    edm::ParameterSetID psID(dg.digest().toString());
    edm::ParameterSet ps(rep, psID);
    pset::Registry::instance()->insertMapped(ps);
  }

  ParameterSetID ParameterSet::emptyParameterSetID() {  // const
    cms::Digest newDigest;
    ParameterSet().toDigest(newDigest);
    return ParameterSetID(newDigest.digest().toString());
  }

  void ParameterSet::copyForModify(ParameterSet const& other) {
    ParameterSet temp(other);
    swap(temp);
    id_ = ParameterSetID();
  }

  void ParameterSet::swap(ParameterSet& other) {
    tbl_.swap(other.tbl_);
    psetTable_.swap(other.psetTable_);
    vpsetTable_.swap(other.vpsetTable_);
    id_.swap(other.id_);
  }

  ParameterSet const& ParameterSet::registerIt() {
    if (!isRegistered()) {
      calculateID();
      pset::Registry::instance()->insertMapped(*this);
    }
    return *this;
  }

  std::unique_ptr<ParameterSet> ParameterSet::popParameterSet(std::string const& name) {
    assert(!isRegistered());
    psettable::iterator it = psetTable_.find(name);
    assert(it != psetTable_.end());
    auto pset = std::make_unique<ParameterSet>();
    std::swap(*pset, it->second.psetForUpdate());
    psetTable_.erase(it);
    return pset;
  }

  void ParameterSet::eraseSimpleParameter(std::string const& name) {
    assert(!isRegistered());
    table::iterator it = tbl_.find(name);
    assert(it != tbl_.end());
    tbl_.erase(it);
  }

  void ParameterSet::eraseOrSetUntrackedParameterSet(std::string const& name) {
    assert(!isRegistered());
    psettable::iterator it = psetTable_.find(name);
    assert(it != psetTable_.end());
    ParameterSet& pset = it->second.psetForUpdate();
    if (pset.isRegistered()) {
      it->second.setIsTracked(false);
    } else {
      psetTable_.erase(it);
    }
  }

  std::vector<ParameterSet> ParameterSet::popVParameterSet(std::string const& name) {
    assert(!isRegistered());
    vpsettable::iterator it = vpsetTable_.find(name);
    assert(it != vpsetTable_.end());
    std::vector<ParameterSet> vpset;
    std::swap(vpset, it->second.vpsetForUpdate());
    vpsetTable_.erase(it);
    return vpset;
  }

  void ParameterSet::calculateID() {
    // make sure contained tracked psets are updated
    for (auto& item : psetTable_) {
      ParameterSet& pset = item.second.psetForUpdate();
      if (!pset.isRegistered()) {
        pset.registerIt();
      }
      item.second.updateID();
    }

    // make sure contained tracked vpsets are updated
    for (vpsettable::iterator i = vpsetTable_.begin(), e = vpsetTable_.end(); i != e; ++i) {
      i->second.registerPsetsAndUpdateIDs();
    }
    //  The old, string base code is found below. Uncomment this and the assert to check whether
    //  there are discrepancies between old and new implementation.
    //    std::string stringrep;
    //    toString(stringrep);
    //    cms::Digest md5alg(stringrep);
    //    id_ = ParameterSetID(md5alg.digest().toString());
    cms::Digest newDigest;
    toDigest(newDigest);
    id_ = ParameterSetID(newDigest.digest().toString());
    //    assert(md5alg.digest().toString() == newDigest.digest().toString());
    assert(isRegistered());
  }

  // ----------------------------------------------------------------------
  // identification
  ParameterSetID ParameterSet::id() const {
    // checks if valid
    if (!isRegistered()) {
      throw Exception(errors::LogicError) << "ParameterSet::id() called prematurely\n"
                                          << "before ParameterSet::registerIt() has been called.\n";
    }
    return id_;
  }

  void ParameterSet::setID(ParameterSetID const& id) { id_ = id; }

  // ----------------------------------------------------------------------
  // Entry-handling
  // ----------------------------------------------------------------------

  Entry const* ParameterSet::getEntryPointerOrThrow_(char const* name) const {
    return getEntryPointerOrThrow_(std::string(name));
  }

  Entry const* ParameterSet::getEntryPointerOrThrow_(std::string const& name) const {
    Entry const* result = retrieveUntracked(name);
    if (result == nullptr)
      throw Exception(errors::Configuration, "MissingParameter:")
          << "The required parameter '" << name << "' was not specified.\n";
    return result;
  }

  template <typename T, typename U>
  T first(std::pair<T, U> const& p) {
    return p.first;
  }

  template <typename T, typename U>
  U second(std::pair<T, U> const& p) {
    return p.second;
  }

  Entry const& ParameterSet::retrieve(char const* name) const { return retrieve(std::string(name)); }

  Entry const& ParameterSet::retrieve(std::string const& name) const {
    table::const_iterator it = tbl_.find(name);
    if (it == tbl_.end()) {
      throw Exception(errors::Configuration, "MissingParameter:") << "Parameter '" << name << "' not found.";
    }
    if (it->second.isTracked() == false) {
      if (name[0] == '@') {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "Framework Error:  Parameter '" << name << "' is incorrectly designated as tracked in the framework.";
      } else {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "Parameter '" << name << "' is designated as tracked in the code,\n"
            << "but is designated as untracked in the configuration file.\n"
            << "Please remove 'untracked' from the configuration file for parameter '" << name << "'.";
      }
    }
    return it->second;
  }  // retrieve()

  Entry const* ParameterSet::retrieveUntracked(char const* name) const { return retrieveUntracked(std::string(name)); }

  Entry const* ParameterSet::retrieveUntracked(std::string const& name) const {
    table::const_iterator it = tbl_.find(name);

    if (it == tbl_.end())
      return nullptr;
    if (it->second.isTracked()) {
      if (name[0] == '@') {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "Framework Error:  Parameter '" << name << "' is incorrectly designated as untracked in the framework.";
      } else {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "Parameter '" << name << "' is designated as untracked in the code,\n"
            << "but is not designated as untracked in the configuration file.\n"
            << "Please change the configuration file to 'untracked <type> " << name << "'.";
      }
    }
    return &it->second;
  }  // retrieve()

  ParameterSetEntry const& ParameterSet::retrieveParameterSet(std::string const& name) const {
    psettable::const_iterator it = psetTable_.find(name);
    if (it == psetTable_.end()) {
      throw Exception(errors::Configuration, "MissingParameter:") << "ParameterSet '" << name << "' not found.";
    }
    if (it->second.isTracked() == false) {
      if (name[0] == '@') {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "Framework Error:  ParameterSet '" << name << "' is incorrectly designated as tracked in the framework.";
      } else {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "ParameterSet '" << name << "' is designated as tracked in the code,\n"
            << "but is designated as untracked in the configuration file.\n"
            << "Please remove 'untracked' from the configuration file for parameter '" << name << "'.";
      }
    }
    return it->second;
  }  // retrieve()

  ParameterSetEntry const* ParameterSet::retrieveUntrackedParameterSet(std::string const& name) const {
    psettable::const_iterator it = psetTable_.find(name);

    if (it == psetTable_.end())
      return nullptr;
    if (it->second.isTracked()) {
      if (name[0] == '@') {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "Framework Error:  ParameterSet '" << name
            << "' is incorrectly designated as untracked in the framework.";
      } else {
        throw Exception(errors::Configuration, "StatusMismatch:")
            << "ParameterSet '" << name << "' is designated as untracked in the code,\n"
            << "but is not designated as untracked in the configuration file.\n"
            << "Please change the configuration file to 'untracked <type> " << name << "'.";
      }
    }
    return &it->second;
  }  // retrieve()

  VParameterSetEntry const& ParameterSet::retrieveVParameterSet(std::string const& name) const {
    vpsettable::const_iterator it = vpsetTable_.find(name);
    if (it == vpsetTable_.end()) {
      throw Exception(errors::Configuration, "MissingParameter:") << "VParameterSet '" << name << "' not found.";
    }
    if (it->second.isTracked() == false) {
      throw Exception(errors::Configuration, "StatusMismatch:")
          << "VParameterSet '" << name << "' is designated as tracked in the code,\n"
          << "but is designated as untracked in the configuration file.\n"
          << "Please remove 'untracked' from the configuration file for parameter '" << name << "'.";
    }
    return it->second;
  }  // retrieve()

  VParameterSetEntry const* ParameterSet::retrieveUntrackedVParameterSet(std::string const& name) const {
    vpsettable::const_iterator it = vpsetTable_.find(name);

    if (it == vpsetTable_.end())
      return nullptr;
    if (it->second.isTracked()) {
      throw Exception(errors::Configuration, "StatusMismatch:")
          << "VParameterSet '" << name << "' is designated as untracked in the code,\n"
          << "but is not designated as untracked in the configuration file.\n"
          << "Please change the configuration file to 'untracked <type> " << name << "'.";
    }
    return &it->second;
  }  // retrieve()

  Entry const* ParameterSet::retrieveUnknown(char const* name) const { return retrieveUnknown(std::string(name)); }

  Entry const* ParameterSet::retrieveUnknown(std::string const& name) const {
    table::const_iterator it = tbl_.find(name);
    if (it == tbl_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  ParameterSetEntry const* ParameterSet::retrieveUnknownParameterSet(std::string const& name) const {
    psettable::const_iterator it = psetTable_.find(name);
    if (it == psetTable_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  VParameterSetEntry const* ParameterSet::retrieveUnknownVParameterSet(std::string const& name) const {
    vpsettable::const_iterator it = vpsetTable_.find(name);
    if (it == vpsetTable_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  std::string ParameterSet::getParameterAsString(std::string const& name) const {
    if (existsAs<ParameterSet>(name)) {
      return retrieveUnknownParameterSet(name)->toString();
    } else if (existsAs<std::vector<ParameterSet> >(name)) {
      return retrieveUnknownVParameterSet(name)->toString();
    } else if (exists(name)) {
      return retrieveUnknown(name)->toString();
    } else {
      throw Exception(errors::Configuration, "getParameterAsString")
          << "Cannot find parameter " << name << " in " << *this;
    }
  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  void ParameterSet::insert(bool okay_to_replace, char const* name, Entry const& value) {
    insert(okay_to_replace, std::string(name), value);
  }

  void ParameterSet::insert(bool okay_to_replace, std::string const& name, Entry const& value) {
    // We should probably get rid of 'okay_to_replace', which will
    // simplify the logic in this function.
    table::iterator it = tbl_.find(name);

    if (it == tbl_.end()) {
      if (!tbl_.insert(std::make_pair(name, value)).second)
        throw Exception(errors::Configuration, "InsertFailure") << "cannot insert " << name << " into a ParameterSet\n";
    } else if (okay_to_replace) {
      it->second = value;
    }
  }  // insert()

  void ParameterSet::insertParameterSet(bool okay_to_replace, std::string const& name, ParameterSetEntry const& entry) {
    // We should probably get rid of 'okay_to_replace', which will
    // simplify the logic in this function.
    psettable::iterator it = psetTable_.find(name);

    if (it == psetTable_.end()) {
      if (!psetTable_.insert(std::make_pair(name, entry)).second)
        throw Exception(errors::Configuration, "InsertFailure") << "cannot insert " << name << " into a ParameterSet\n";
    } else if (okay_to_replace) {
      it->second = entry;
    }
  }  // insert()

  void ParameterSet::insertVParameterSet(bool okay_to_replace,
                                         std::string const& name,
                                         VParameterSetEntry const& entry) {
    // We should probably get rid of 'okay_to_replace', which will
    // simplify the logic in this function.
    vpsettable::iterator it = vpsetTable_.find(name);

    if (it == vpsetTable_.end()) {
      if (!vpsetTable_.insert(std::make_pair(name, entry)).second)
        throw Exception(errors::Configuration, "InsertFailure")
            << "cannot insert " << name << " into a VParameterSet\n";
    } else if (okay_to_replace) {
      it->second = entry;
    }
  }  // insert()

  void ParameterSet::augment(ParameterSet const& from) {
    // This preemptive invalidation may be more agressive than necessary.
    invalidateRegistration(std::string());
    if (&from == this) {
      return;
    }

    for (table::const_iterator b = from.tbl_.begin(), e = from.tbl_.end(); b != e; ++b) {
      this->insert(false, b->first, b->second);
    }
    for (psettable::const_iterator b = from.psetTable_.begin(), e = from.psetTable_.end(); b != e; ++b) {
      this->insertParameterSet(false, b->first, b->second);
    }
    for (vpsettable::const_iterator b = from.vpsetTable_.begin(), e = from.vpsetTable_.end(); b != e; ++b) {
      this->insertVParameterSet(false, b->first, b->second);
    }
  }  // augment()

  void ParameterSet::copyFrom(ParameterSet const& from, std::string const& name) {
    invalidateRegistration(std::string());
    if (from.existsAs<ParameterSet>(name)) {
      this->insertParameterSet(false, name, *(from.retrieveUnknownParameterSet(name)));
    } else if (from.existsAs<std::vector<ParameterSet> >(name)) {
      this->insertVParameterSet(false, name, *(from.retrieveUnknownVParameterSet(name)));
    } else if (from.exists(name)) {
      this->insert(false, name, *(from.retrieveUnknown(name)));
    } else {
      throw Exception(errors::Configuration, "copyFrom") << "Cannot find parameter " << name << " in " << from;
    }
  }

  ParameterSet* ParameterSet::getPSetForUpdate(std::string const& name, bool& isTracked) {
    assert(!isRegistered());
    isTracked = false;
    psettable::iterator it = psetTable_.find(name);
    if (it == psetTable_.end())
      return nullptr;
    isTracked = it->second.isTracked();
    return &it->second.psetForUpdate();
  }

  VParameterSetEntry* ParameterSet::getPSetVectorForUpdate(std::string const& name) {
    assert(!isRegistered());
    vpsettable::iterator it = vpsetTable_.find(name);
    if (it == vpsetTable_.end())
      return nullptr;
    return &it->second;
  }

  // ----------------------------------------------------------------------
  // coding
  // ----------------------------------------------------------------------

  void ParameterSet::toString(std::string& rep) const { toStringImp(rep, false); }

  void ParameterSet::allToString(std::string& rep) const { toStringImp(rep, true); }

  void ParameterSet::toStringImp(std::string& rep, bool useAll) const {
    // make sure the PSets get filled
    if (empty()) {
      rep += "<>";
      return;
    }
    size_t size = 1;
    for (table::const_iterator b = tbl_.begin(), e = tbl_.end(); b != e; ++b) {
      if (useAll || b->second.isTracked()) {
        size += 2;
        size += b->first.size();
        size += b->second.sizeOfString();
      }
    }
    for (psettable::const_iterator b = psetTable_.begin(), e = psetTable_.end(); b != e; ++b) {
      if (useAll || b->second.isTracked()) {
        size += 2;
        size += b->first.size();
        size += b->first.size();
        size += b->first.size();
        size += sizeof(ParameterSetID);
      }
    }
    for (vpsettable::const_iterator b = vpsetTable_.begin(), e = vpsetTable_.end(); b != e; ++b) {
      if (useAll || b->second.isTracked()) {
        size += 2;
        size += b->first.size();
        size += sizeof(ParameterSetID) * b->second.vpset().size();
      }
    }

    rep.reserve(rep.size() + size);
    rep += '<';
    std::string start;
    std::string const between(";");
    for (table::const_iterator b = tbl_.begin(), e = tbl_.end(); b != e; ++b) {
      if (useAll || b->second.isTracked()) {
        rep += start;
        rep += b->first;
        rep += '=';
        b->second.toString(rep);
        start = between;
      }
    }
    for (psettable::const_iterator b = psetTable_.begin(), e = psetTable_.end(); b != e; ++b) {
      if (useAll || b->second.isTracked()) {
        rep += start;
        rep += b->first;
        rep += '=';
        b->second.toString(rep);
        start = between;
      }
    }
    for (vpsettable::const_iterator b = vpsetTable_.begin(), e = vpsetTable_.end(); b != e; ++b) {
      if (useAll || b->second.isTracked()) {
        rep += start;
        rep += b->first;
        rep += '=';
        b->second.toString(rep);
        start = between;
      }
    }

    rep += '>';
  }  // to_string()

  void ParameterSet::toDigest(cms::Digest& digest) const {
    digest.append("<", 1);
    bool started = false;
    for (table::const_iterator b = tbl_.begin(), e = tbl_.end(); b != e; ++b) {
      if (b->second.isTracked()) {
        if (started)
          digest.append(";", 1);
        digest.append(b->first);
        digest.append("=", 1);
        b->second.toDigest(digest);
        started = true;
      }
    }
    for (psettable::const_iterator b = psetTable_.begin(), e = psetTable_.end(); b != e; ++b) {
      if (b->second.isTracked()) {
        if (started)
          digest.append(";", 1);
        digest.append(b->first);
        digest.append("=", 1);
        b->second.toDigest(digest);
        started = true;
      }
    }
    for (vpsettable::const_iterator b = vpsetTable_.begin(), e = vpsetTable_.end(); b != e; ++b) {
      if (b->second.isTracked()) {
        if (started)
          digest.append(";", 1);
        digest.append(b->first);
        digest.append("=", 1);
        b->second.toDigest(digest);
      }
    }

    digest.append(">", 1);
  }

  std::string ParameterSet::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  // ----------------------------------------------------------------------

  bool ParameterSet::fromString(std::string const& from) {
    std::vector<std::string> temp;
    if (!split(std::back_inserter(temp), from, '<', ';', '>'))
      return false;

    tbl_.clear();  // precaution
    for (std::vector<std::string>::const_iterator b = temp.begin(), e = temp.end(); b != e; ++b) {
      // locate required name/value separator
      std::string::const_iterator q = find_in_all(*b, '=');
      if (q == b->end())
        return false;

      // form name unique to this ParameterSet
      std::string name = std::string(b->begin(), q);
      if (tbl_.find(name) != tbl_.end())
        return false;

      std::string rep(q + 1, b->end());
      // entries are generically of the form tracked-type-rep
      if (rep[0] == '-') {
      }
      if (rep[1] == 'Q') {
        ParameterSetEntry psetEntry(rep);
        if (!psetTable_.insert(std::make_pair(name, psetEntry)).second) {
          return false;
        }
      } else if (rep[1] == 'q') {
        VParameterSetEntry vpsetEntry(rep);
        if (!vpsetTable_.insert(std::make_pair(name, vpsetEntry)).second) {
          return false;
        }
      } else if (rep[1] == 'P') {
        Entry value(name, rep);
        ParameterSetEntry psetEntry(value.getPSet(), value.isTracked());
        if (!psetTable_.insert(std::make_pair(name, psetEntry)).second) {
          return false;
        }
      } else if (rep[1] == 'p') {
        Entry value(name, rep);
        VParameterSetEntry vpsetEntry(value.getVPSet(), value.isTracked());
        if (!vpsetTable_.insert(std::make_pair(name, vpsetEntry)).second) {
          return false;
        }
      } else {
        // form value and insert name/value pair
        Entry value(name, rep);
        if (!tbl_.insert(std::make_pair(name, value)).second) {
          return false;
        }
      }
    }

    return true;
  }  // from_string()

  std::vector<FileInPath>::size_type ParameterSet::getAllFileInPaths(std::vector<FileInPath>& output) const {
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

  std::vector<std::string> ParameterSet::getParameterNames() const {
    using std::placeholders::_1;
    std::vector<std::string> returnValue;
    std::transform(tbl_.begin(),
                   tbl_.end(),
                   back_inserter(returnValue),
                   std::bind(&std::pair<std::string const, Entry>::first, _1));
    std::transform(psetTable_.begin(),
                   psetTable_.end(),
                   back_inserter(returnValue),
                   std::bind(&std::pair<std::string const, ParameterSetEntry>::first, _1));
    std::transform(vpsetTable_.begin(),
                   vpsetTable_.end(),
                   back_inserter(returnValue),
                   std::bind(&std::pair<std::string const, VParameterSetEntry>::first, _1));
    return returnValue;
  }

  bool ParameterSet::exists(std::string const& parameterName) const {
    return (tbl_.find(parameterName) != tbl_.end() || psetTable_.find(parameterName) != psetTable_.end() ||
            vpsetTable_.find(parameterName) != vpsetTable_.end());
  }

  ParameterSet ParameterSet::trackedPart() const {
    ParameterSet result;
    for (table::const_iterator tblItr = tbl_.begin(); tblItr != tbl_.end(); ++tblItr) {
      if (tblItr->second.isTracked()) {
        result.tbl_.insert(*tblItr);
      }
    }
    for (psettable::const_iterator psetItr = psetTable_.begin(); psetItr != psetTable_.end(); ++psetItr) {
      if (psetItr->second.isTracked()) {
        result.addParameter<ParameterSet>(psetItr->first, psetItr->second.pset().trackedPart());
      }
    }
    for (vpsettable::const_iterator vpsetItr = vpsetTable_.begin(); vpsetItr != vpsetTable_.end(); ++vpsetItr) {
      if (vpsetItr->second.isTracked()) {
        VParameterSet vresult;
        std::vector<ParameterSet> const& this_vpset = vpsetItr->second.vpset();

        typedef std::vector<ParameterSet>::const_iterator Iter;
        for (Iter i = this_vpset.begin(), e = this_vpset.end(); i != e; ++i) {
          vresult.push_back(i->trackedPart());
        }
        result.addParameter<VParameterSet>(vpsetItr->first, vresult);
      }
    }
    return result;
  }

  /*
  // Comment out unneeded function
  size_t
  ParameterSet::getAllParameterSetNames(std::vector<std::string>& output) const {
    using std::placeholders::_1;
    std::transform(psetTable_.begin(), psetTable_.end(), back_inserter(output),
                   std::bind(&std::pair<std::string const, ParameterSetEntry>::first, _1));
    return output.size();
  }
*/

  size_t ParameterSet::getParameterSetNames(std::vector<std::string>& output, bool trackiness) const {
    for (psettable::const_iterator psetItr = psetTable_.begin(); psetItr != psetTable_.end(); ++psetItr) {
      if (psetItr->second.isTracked() == trackiness) {
        output.push_back(psetItr->first);
      }
    }
    return output.size();
  }

  size_t ParameterSet::getParameterSetVectorNames(std::vector<std::string>& output, bool trackiness) const {
    for (vpsettable::const_iterator vpsetItr = vpsetTable_.begin(); vpsetItr != vpsetTable_.end(); ++vpsetItr) {
      if (vpsetItr->second.isTracked() == trackiness) {
        output.push_back(vpsetItr->first);
      }
    }
    return output.size();
  }

  size_t ParameterSet::getNamesByCode_(char code, bool trackiness, std::vector<std::string>& output) const {
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
      if (e.typeCode() == code && e.isTracked() == trackiness) {  // if it is a vector of ParameterSet
        ++count;
        output.push_back(it->first);  // save the name
      }
      ++it;
    }
    return count;
  }

  template <>
  std::vector<std::string> ParameterSet::getParameterNamesForType<FileInPath>(bool trackiness) const {
    std::vector<std::string> result;
    getNamesByCode_('F', trackiness, result);
    return result;
  }

  bool operator==(ParameterSet const& a, ParameterSet const& b) {
    if (a.isRegistered() && b.isRegistered()) {
      return (a.id() == b.id());
    }
    return isTransientEqual(a.trackedPart(), b.trackedPart());
  }

  bool isTransientEqual(ParameterSet const& a, ParameterSet const& b) {
    if (a.tbl().size() != b.tbl().size()) {
      return false;
    }
    if (a.psetTable().size() != b.psetTable().size()) {
      return false;
    }
    if (a.vpsetTable().size() != b.vpsetTable().size()) {
      return false;
    }
    typedef ParameterSet::table::const_iterator Ti;
    for (Ti i = a.tbl().begin(), e = a.tbl().end(), j = b.tbl().begin(), f = b.tbl().end(); i != e && j != f;
         ++i, ++j) {
      if (*i != *j) {
        return false;
      }
    }
    typedef ParameterSet::psettable::const_iterator Pi;
    for (Pi i = a.psetTable().begin(), e = a.psetTable().end(), j = b.psetTable().begin(), f = b.psetTable().end();
         i != e && j != f;
         ++i, ++j) {
      if (i->first != j->first) {
        return false;
      }
      if (i->second.isTracked() != j->second.isTracked()) {
        return false;
      }
      if (!isTransientEqual(i->second.pset(), j->second.pset())) {
        return false;
      }
    }
    typedef ParameterSet::vpsettable::const_iterator PVi;
    for (PVi i = a.vpsetTable().begin(), e = a.vpsetTable().end(), j = b.vpsetTable().begin(), f = b.vpsetTable().end();
         i != e && j != f;
         ++i, ++j) {
      if (i->first != j->first) {
        return false;
      }
      if (i->second.isTracked() != j->second.isTracked()) {
        return false;
      }
      std::vector<ParameterSet> const& iv = i->second.vpset();
      std::vector<ParameterSet> const& jv = j->second.vpset();
      if (iv.size() != jv.size()) {
        return false;
      }
      for (size_t k = 0; k < iv.size(); ++k) {
        if (!isTransientEqual(iv[k], jv[k])) {
          return false;
        }
      }
    }
    return true;
  }

  std::string ParameterSet::dump(unsigned int indent) const {
    std::ostringstream os;
    // indent a bit
    std::string indentation(indent, ' ');
    os << "{" << std::endl;
    for (table::const_iterator i = tbl_.begin(), e = tbl_.end(); i != e; ++i) {
      os << indentation << "  " << i->first << ": " << i->second << std::endl;
    }
    for (psettable::const_iterator i = psetTable_.begin(), e = psetTable_.end(); i != e; ++i) {
      // indent a bit
      std::string n = i->first;
      ParameterSetEntry const& pe = i->second;
      os << indentation << "  " << n << ": " << pe.dump(indent + 2) << std::endl;
    }
    for (vpsettable::const_iterator i = vpsetTable_.begin(), e = vpsetTable_.end(); i != e; ++i) {
      // indent a bit
      std::string n = i->first;
      VParameterSetEntry const& pe = i->second;
      os << indentation << "  " << n << ": " << pe.dump(indent + 2) << std::endl;
    }
    os << indentation << "}";
    return os.str();
  }

  std::ostream& operator<<(std::ostream& os, ParameterSet const& pset) {
    os << pset.dump();
    return os;
  }

  // Free function to return a parameterSet given its ID.
  ParameterSet const& getParameterSet(ParameterSetID const& id) {
    ParameterSet const* result = nullptr;
    if (nullptr == (result = pset::Registry::instance()->getMapped(id))) {
      throw Exception(errors::LogicError, "MissingParameterSet:") << "Parameter Set ID '" << id << "' not found.";
    }
    return *result;
  }

  ParameterSet const& getProcessParameterSetContainingModule(ModuleDescription const& moduleDescription) {
    return getParameterSet(moduleDescription.mainParameterSetID());
  }

  void ParameterSet::deprecatedInputTagWarning(std::string const& name, std::string const& label) const {
    LogWarning("Configuration") << "Warning:\n\tstring " << name << " = \"" << label << "\"\nis deprecated, "
                                << "please update your config file to use\n\tInputTag " << name << " = " << label;
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool

  template <>
  bool ParameterSet::getParameter<bool>(std::string const& name) const {
    return retrieve(name).getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template <>
  int ParameterSet::getParameter<int>(std::string const& name) const {
    return retrieve(name).getInt32();
  }

  template <>
  std::vector<int> ParameterSet::getParameter<std::vector<int> >(std::string const& name) const {
    return retrieve(name).getVInt32();
  }

  // ----------------------------------------------------------------------
  // Int64, vInt64

  template <>
  long long ParameterSet::getParameter<long long>(std::string const& name) const {
    return retrieve(name).getInt64();
  }

  template <>
  std::vector<long long> ParameterSet::getParameter<std::vector<long long> >(std::string const& name) const {
    return retrieve(name).getVInt64();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template <>
  unsigned int ParameterSet::getParameter<unsigned int>(std::string const& name) const {
    return retrieve(name).getUInt32();
  }

  template <>
  std::vector<unsigned int> ParameterSet::getParameter<std::vector<unsigned int> >(std::string const& name) const {
    return retrieve(name).getVUInt32();
  }

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template <>
  unsigned long long ParameterSet::getParameter<unsigned long long>(std::string const& name) const {
    return retrieve(name).getUInt64();
  }

  template <>
  std::vector<unsigned long long> ParameterSet::getParameter<std::vector<unsigned long long> >(
      std::string const& name) const {
    return retrieve(name).getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble

  template <>
  double ParameterSet::getParameter<double>(std::string const& name) const {
    return retrieve(name).getDouble();
  }

  template <>
  std::vector<double> ParameterSet::getParameter<std::vector<double> >(std::string const& name) const {
    return retrieve(name).getVDouble();
  }

  // ----------------------------------------------------------------------
  // String, vString

  template <>
  std::string ParameterSet::getParameter<std::string>(std::string const& name) const {
    return retrieve(name).getString();
  }

  template <>
  std::vector<std::string> ParameterSet::getParameter<std::vector<std::string> >(std::string const& name) const {
    return retrieve(name).getVString();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  template <>
  FileInPath ParameterSet::getParameter<FileInPath>(std::string const& name) const {
    return retrieve(name).getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag

  template <>
  InputTag ParameterSet::getParameter<InputTag>(std::string const& name) const {
    Entry const& e_input = retrieve(name);
    switch (e_input.typeCode()) {
      case 't':  // InputTag
        return e_input.getInputTag();
      case 'S':  // string
        std::string const& label = e_input.getString();
        deprecatedInputTagWarning(name, label);
        return InputTag(label);
    }
    throw Exception(errors::Configuration, "ValueError")
        << "type of " << name << " is expected to be InputTag or string (deprecated)";
  }

  // ----------------------------------------------------------------------
  // VInputTag

  template <>
  std::vector<InputTag> ParameterSet::getParameter<std::vector<InputTag> >(std::string const& name) const {
    return retrieve(name).getVInputTag();
  }

  // ----------------------------------------------------------------------
  // ESInputTag

  template <>
  ESInputTag ParameterSet::getParameter<ESInputTag>(std::string const& name) const {
    return retrieve(name).getESInputTag();
  }

  // ----------------------------------------------------------------------
  // VESInputTag

  template <>
  std::vector<ESInputTag> ParameterSet::getParameter<std::vector<ESInputTag> >(std::string const& name) const {
    return retrieve(name).getVESInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID

  template <>
  EventID ParameterSet::getParameter<EventID>(std::string const& name) const {
    return retrieve(name).getEventID();
  }

  // ----------------------------------------------------------------------
  // VEventID

  template <>
  std::vector<EventID> ParameterSet::getParameter<std::vector<EventID> >(std::string const& name) const {
    return retrieve(name).getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template <>
  LuminosityBlockID ParameterSet::getParameter<LuminosityBlockID>(std::string const& name) const {
    return retrieve(name).getLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template <>
  std::vector<LuminosityBlockID> ParameterSet::getParameter<std::vector<LuminosityBlockID> >(
      std::string const& name) const {
    return retrieve(name).getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // EventRange

  template <>
  EventRange ParameterSet::getParameter<EventRange>(std::string const& name) const {
    return retrieve(name).getEventRange();
  }

  // ----------------------------------------------------------------------
  // VEventRange

  template <>
  std::vector<EventRange> ParameterSet::getParameter<std::vector<EventRange> >(std::string const& name) const {
    return retrieve(name).getVEventRange();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockRange

  template <>
  LuminosityBlockRange ParameterSet::getParameter<LuminosityBlockRange>(std::string const& name) const {
    return retrieve(name).getLuminosityBlockRange();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  template <>
  std::vector<LuminosityBlockRange> ParameterSet::getParameter<std::vector<LuminosityBlockRange> >(
      std::string const& name) const {
    return retrieve(name).getVLuminosityBlockRange();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet

  template <>
  ParameterSet ParameterSet::getParameter<ParameterSet>(std::string const& name) const {
    return getParameterSet(name);
  }

  template <>
  VParameterSet ParameterSet::getParameter<VParameterSet>(std::string const& name) const {
    return getParameterSetVector(name);
  }

  // untracked parameters

  // ----------------------------------------------------------------------
  // Bool, vBool

  template <>
  bool ParameterSet::getUntrackedParameter<bool>(std::string const& name, bool const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getBool();
  }

  template <>
  bool ParameterSet::getUntrackedParameter<bool>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template <>
  int ParameterSet::getUntrackedParameter<int>(std::string const& name, int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getInt32();
  }

  template <>
  int ParameterSet::getUntrackedParameter<int>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInt32();
  }

  template <>
  std::vector<int> ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name,
                                                                          std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVInt32();
  }

  template <>
  std::vector<int> ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInt32();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template <>
  unsigned int ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name,
                                                                 unsigned int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getUInt32();
  }

  template <>
  unsigned int ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getUInt32();
  }

  template <>
  std::vector<unsigned int> ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(
      std::string const& name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVUInt32();
  }

  template <>
  std::vector<unsigned int> ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(
      std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVUInt32();
  }

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template <>
  unsigned long long ParameterSet::getUntrackedParameter<unsigned long long>(
      std::string const& name, unsigned long long const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getUInt64();
  }

  template <>
  unsigned long long ParameterSet::getUntrackedParameter<unsigned long long>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getUInt64();
  }

  template <>
  std::vector<unsigned long long> ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(
      std::string const& name, std::vector<unsigned long long> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVUInt64();
  }

  template <>
  std::vector<unsigned long long> ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(
      std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Int64, Vint64

  template <>
  long long ParameterSet::getUntrackedParameter<long long>(std::string const& name,
                                                           long long const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getInt64();
  }

  template <>
  long long ParameterSet::getUntrackedParameter<long long>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInt64();
  }

  template <>
  std::vector<long long> ParameterSet::getUntrackedParameter<std::vector<long long> >(
      std::string const& name, std::vector<long long> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVInt64();
  }

  template <>
  std::vector<long long> ParameterSet::getUntrackedParameter<std::vector<long long> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble

  template <>
  double ParameterSet::getUntrackedParameter<double>(std::string const& name, double const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getDouble();
  }

  template <>
  double ParameterSet::getUntrackedParameter<double>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getDouble();
  }

  template <>
  std::vector<double> ParameterSet::getUntrackedParameter<std::vector<double> >(
      std::string const& name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVDouble();
  }

  template <>
  std::vector<double> ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVDouble();
  }

  // ----------------------------------------------------------------------
  // String, vString

  template <>
  std::string ParameterSet::getUntrackedParameter<std::string>(std::string const& name,
                                                               std::string const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getString();
  }

  template <>
  std::string ParameterSet::getUntrackedParameter<std::string>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getString();
  }

  template <>
  std::vector<std::string> ParameterSet::getUntrackedParameter<std::vector<std::string> >(
      std::string const& name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVString();
  }

  template <>
  std::vector<std::string> ParameterSet::getUntrackedParameter<std::vector<std::string> >(
      std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVString();
  }

  // ----------------------------------------------------------------------
  //  FileInPath

  template <>
  FileInPath ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name,
                                                             FileInPath const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getFileInPath();
  }

  template <>
  FileInPath ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template <>
  InputTag ParameterSet::getUntrackedParameter<InputTag>(std::string const& name, InputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getInputTag();
  }

  template <>
  InputTag ParameterSet::getUntrackedParameter<InputTag>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInputTag();
  }

  template <>
  std::vector<InputTag> ParameterSet::getUntrackedParameter<std::vector<InputTag> >(
      std::string const& name, std::vector<InputTag> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVInputTag();
  }

  template <>
  std::vector<InputTag> ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInputTag();
  }

  // ----------------------------------------------------------------------
  // ESInputTag, VESInputTag

  template <>
  ESInputTag ParameterSet::getUntrackedParameter<ESInputTag>(std::string const& name,
                                                             ESInputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getESInputTag();
  }

  template <>
  ESInputTag ParameterSet::getUntrackedParameter<ESInputTag>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getESInputTag();
  }

  template <>
  std::vector<ESInputTag> ParameterSet::getUntrackedParameter<std::vector<ESInputTag> >(
      std::string const& name, std::vector<ESInputTag> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVESInputTag();
  }

  template <>
  std::vector<ESInputTag> ParameterSet::getUntrackedParameter<std::vector<ESInputTag> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVESInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template <>
  EventID ParameterSet::getUntrackedParameter<EventID>(std::string const& name, EventID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getEventID();
  }

  template <>
  EventID ParameterSet::getUntrackedParameter<EventID>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getEventID();
  }

  template <>
  std::vector<EventID> ParameterSet::getUntrackedParameter<std::vector<EventID> >(
      std::string const& name, std::vector<EventID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVEventID();
  }

  template <>
  std::vector<EventID> ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template <>
  LuminosityBlockID ParameterSet::getUntrackedParameter<LuminosityBlockID>(
      std::string const& name, LuminosityBlockID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getLuminosityBlockID();
  }

  template <>
  LuminosityBlockID ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockID();
  }

  template <>
  std::vector<LuminosityBlockID> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(
      std::string const& name, std::vector<LuminosityBlockID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVLuminosityBlockID();
  }

  template <>
  std::vector<LuminosityBlockID> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(
      std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // EventRange, VEventRange

  template <>
  EventRange ParameterSet::getUntrackedParameter<EventRange>(std::string const& name,
                                                             EventRange const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getEventRange();
  }

  template <>
  EventRange ParameterSet::getUntrackedParameter<EventRange>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getEventRange();
  }

  template <>
  std::vector<EventRange> ParameterSet::getUntrackedParameter<std::vector<EventRange> >(
      std::string const& name, std::vector<EventRange> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVEventRange();
  }

  template <>
  std::vector<EventRange> ParameterSet::getUntrackedParameter<std::vector<EventRange> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVEventRange();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockRange, VLuminosityBlockRange

  template <>
  LuminosityBlockRange ParameterSet::getUntrackedParameter<LuminosityBlockRange>(
      std::string const& name, LuminosityBlockRange const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getLuminosityBlockRange();
  }

  template <>
  LuminosityBlockRange ParameterSet::getUntrackedParameter<LuminosityBlockRange>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockRange();
  }

  template <>
  std::vector<LuminosityBlockRange> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(
      std::string const& name, std::vector<LuminosityBlockRange> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVLuminosityBlockRange();
  }

  template <>
  std::vector<LuminosityBlockRange> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(
      std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockRange();
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool

  template <>
  bool ParameterSet::getParameter<bool>(char const* name) const {
    return retrieve(name).getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template <>
  int ParameterSet::getParameter<int>(char const* name) const {
    return retrieve(name).getInt32();
  }

  template <>
  std::vector<int> ParameterSet::getParameter<std::vector<int> >(char const* name) const {
    return retrieve(name).getVInt32();
  }

  // ----------------------------------------------------------------------
  // Int64, vInt64

  template <>
  long long ParameterSet::getParameter<long long>(char const* name) const {
    return retrieve(name).getInt64();
  }

  template <>
  std::vector<long long> ParameterSet::getParameter<std::vector<long long> >(char const* name) const {
    return retrieve(name).getVInt64();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template <>
  unsigned int ParameterSet::getParameter<unsigned int>(char const* name) const {
    return retrieve(name).getUInt32();
  }

  template <>
  std::vector<unsigned int> ParameterSet::getParameter<std::vector<unsigned int> >(char const* name) const {
    return retrieve(name).getVUInt32();
  }

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template <>
  unsigned long long ParameterSet::getParameter<unsigned long long>(char const* name) const {
    return retrieve(name).getUInt64();
  }

  template <>
  std::vector<unsigned long long> ParameterSet::getParameter<std::vector<unsigned long long> >(char const* name) const {
    return retrieve(name).getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble

  template <>
  double ParameterSet::getParameter<double>(char const* name) const {
    return retrieve(name).getDouble();
  }

  template <>
  std::vector<double> ParameterSet::getParameter<std::vector<double> >(char const* name) const {
    return retrieve(name).getVDouble();
  }

  // ----------------------------------------------------------------------
  // String, vString

  template <>
  std::string ParameterSet::getParameter<std::string>(char const* name) const {
    return retrieve(name).getString();
  }

  template <>
  std::vector<std::string> ParameterSet::getParameter<std::vector<std::string> >(char const* name) const {
    return retrieve(name).getVString();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  template <>
  FileInPath ParameterSet::getParameter<FileInPath>(char const* name) const {
    return retrieve(name).getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag

  template <>
  InputTag ParameterSet::getParameter<InputTag>(char const* name) const {
    Entry const& e_input = retrieve(name);
    switch (e_input.typeCode()) {
      case 't':  // InputTag
        return e_input.getInputTag();
      case 'S':  // string
        std::string const& label = e_input.getString();
        deprecatedInputTagWarning(name, label);
        return InputTag(label);
    }
    throw Exception(errors::Configuration, "ValueError")
        << "type of " << name << " is expected to be InputTag or string (deprecated)";
  }

  // ----------------------------------------------------------------------
  // VInputTag

  template <>
  std::vector<InputTag> ParameterSet::getParameter<std::vector<InputTag> >(char const* name) const {
    return retrieve(name).getVInputTag();
  }

  // ----------------------------------------------------------------------
  // ESInputTag

  template <>
  ESInputTag ParameterSet::getParameter<ESInputTag>(char const* name) const {
    return retrieve(name).getESInputTag();
  }

  // ----------------------------------------------------------------------
  // VESInputTag

  template <>
  std::vector<ESInputTag> ParameterSet::getParameter<std::vector<ESInputTag> >(char const* name) const {
    return retrieve(name).getVESInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID

  template <>
  EventID ParameterSet::getParameter<EventID>(char const* name) const {
    return retrieve(name).getEventID();
  }

  // ----------------------------------------------------------------------
  // VEventID

  template <>
  std::vector<EventID> ParameterSet::getParameter<std::vector<EventID> >(char const* name) const {
    return retrieve(name).getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template <>
  LuminosityBlockID ParameterSet::getParameter<LuminosityBlockID>(char const* name) const {
    return retrieve(name).getLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template <>
  std::vector<LuminosityBlockID> ParameterSet::getParameter<std::vector<LuminosityBlockID> >(char const* name) const {
    return retrieve(name).getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // EventRange

  template <>
  EventRange ParameterSet::getParameter<EventRange>(char const* name) const {
    return retrieve(name).getEventRange();
  }

  // ----------------------------------------------------------------------
  // VEventRange

  template <>
  std::vector<EventRange> ParameterSet::getParameter<std::vector<EventRange> >(char const* name) const {
    return retrieve(name).getVEventRange();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockRange

  template <>
  LuminosityBlockRange ParameterSet::getParameter<LuminosityBlockRange>(char const* name) const {
    return retrieve(name).getLuminosityBlockRange();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  template <>
  std::vector<LuminosityBlockRange> ParameterSet::getParameter<std::vector<LuminosityBlockRange> >(
      char const* name) const {
    return retrieve(name).getVLuminosityBlockRange();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet

  template <>
  ParameterSet ParameterSet::getParameter<ParameterSet>(char const* name) const {
    return getParameterSet(name);
  }

  template <>
  VParameterSet ParameterSet::getParameter<VParameterSet>(char const* name) const {
    return getParameterSetVector(name);
  }

  // untracked parameters

  // ----------------------------------------------------------------------
  // Bool, vBool

  template <>
  bool ParameterSet::getUntrackedParameter<bool>(char const* name, bool const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getBool();
  }

  template <>
  bool ParameterSet::getUntrackedParameter<bool>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template <>
  int ParameterSet::getUntrackedParameter<int>(char const* name, int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getInt32();
  }

  template <>
  int ParameterSet::getUntrackedParameter<int>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInt32();
  }

  template <>
  std::vector<int> ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name,
                                                                          std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVInt32();
  }

  template <>
  std::vector<int> ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInt32();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template <>
  unsigned int ParameterSet::getUntrackedParameter<unsigned int>(char const* name,
                                                                 unsigned int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getUInt32();
  }

  template <>
  unsigned int ParameterSet::getUntrackedParameter<unsigned int>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getUInt32();
  }

  template <>
  std::vector<unsigned int> ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(
      char const* name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVUInt32();
  }

  template <>
  std::vector<unsigned int> ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVUInt32();
  }

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template <>
  unsigned long long ParameterSet::getUntrackedParameter<unsigned long long>(
      char const* name, unsigned long long const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getUInt64();
  }

  template <>
  unsigned long long ParameterSet::getUntrackedParameter<unsigned long long>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getUInt64();
  }

  template <>
  std::vector<unsigned long long> ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(
      char const* name, std::vector<unsigned long long> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVUInt64();
  }

  template <>
  std::vector<unsigned long long> ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(
      char const* name) const {
    return getEntryPointerOrThrow_(name)->getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Int64, Vint64

  template <>
  long long ParameterSet::getUntrackedParameter<long long>(char const* name, long long const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getInt64();
  }

  template <>
  long long ParameterSet::getUntrackedParameter<long long>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInt64();
  }

  template <>
  std::vector<long long> ParameterSet::getUntrackedParameter<std::vector<long long> >(
      char const* name, std::vector<long long> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVInt64();
  }

  template <>
  std::vector<long long> ParameterSet::getUntrackedParameter<std::vector<long long> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble

  template <>
  double ParameterSet::getUntrackedParameter<double>(char const* name, double const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getDouble();
  }

  template <>
  double ParameterSet::getUntrackedParameter<double>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getDouble();
  }

  template <>
  std::vector<double> ParameterSet::getUntrackedParameter<std::vector<double> >(
      char const* name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVDouble();
  }

  template <>
  std::vector<double> ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVDouble();
  }

  // ----------------------------------------------------------------------
  // String, vString

  template <>
  std::string ParameterSet::getUntrackedParameter<std::string>(char const* name,
                                                               std::string const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getString();
  }

  template <>
  std::string ParameterSet::getUntrackedParameter<std::string>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getString();
  }

  template <>
  std::vector<std::string> ParameterSet::getUntrackedParameter<std::vector<std::string> >(
      char const* name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVString();
  }

  template <>
  std::vector<std::string> ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVString();
  }

  // ----------------------------------------------------------------------
  //  FileInPath

  template <>
  FileInPath ParameterSet::getUntrackedParameter<FileInPath>(char const* name, FileInPath const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getFileInPath();
  }

  template <>
  FileInPath ParameterSet::getUntrackedParameter<FileInPath>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template <>
  InputTag ParameterSet::getUntrackedParameter<InputTag>(char const* name, InputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getInputTag();
  }

  template <>
  InputTag ParameterSet::getUntrackedParameter<InputTag>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInputTag();
  }

  template <>
  std::vector<InputTag> ParameterSet::getUntrackedParameter<std::vector<InputTag> >(
      char const* name, std::vector<InputTag> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVInputTag();
  }

  template <>
  std::vector<InputTag> ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInputTag();
  }

  // ----------------------------------------------------------------------
  // ESInputTag, VESInputTag

  template <>
  ESInputTag ParameterSet::getUntrackedParameter<ESInputTag>(char const* name, ESInputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getESInputTag();
  }

  template <>
  ESInputTag ParameterSet::getUntrackedParameter<ESInputTag>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getESInputTag();
  }

  template <>
  std::vector<ESInputTag> ParameterSet::getUntrackedParameter<std::vector<ESInputTag> >(
      char const* name, std::vector<ESInputTag> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVESInputTag();
  }

  template <>
  std::vector<ESInputTag> ParameterSet::getUntrackedParameter<std::vector<ESInputTag> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVESInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template <>
  EventID ParameterSet::getUntrackedParameter<EventID>(char const* name, EventID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getEventID();
  }

  template <>
  EventID ParameterSet::getUntrackedParameter<EventID>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getEventID();
  }

  template <>
  std::vector<EventID> ParameterSet::getUntrackedParameter<std::vector<EventID> >(
      char const* name, std::vector<EventID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVEventID();
  }

  template <>
  std::vector<EventID> ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template <>
  LuminosityBlockID ParameterSet::getUntrackedParameter<LuminosityBlockID>(
      char const* name, LuminosityBlockID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getLuminosityBlockID();
  }

  template <>
  LuminosityBlockID ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockID();
  }

  template <>
  std::vector<LuminosityBlockID> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(
      char const* name, std::vector<LuminosityBlockID> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVLuminosityBlockID();
  }

  template <>
  std::vector<LuminosityBlockID> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(
      char const* name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // EventRange, VEventRange

  template <>
  EventRange ParameterSet::getUntrackedParameter<EventRange>(char const* name, EventRange const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getEventRange();
  }

  template <>
  EventRange ParameterSet::getUntrackedParameter<EventRange>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getEventRange();
  }

  template <>
  std::vector<EventRange> ParameterSet::getUntrackedParameter<std::vector<EventRange> >(
      char const* name, std::vector<EventRange> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVEventRange();
  }

  template <>
  std::vector<EventRange> ParameterSet::getUntrackedParameter<std::vector<EventRange> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVEventRange();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockRange, VLuminosityBlockRange

  template <>
  LuminosityBlockRange ParameterSet::getUntrackedParameter<LuminosityBlockRange>(
      char const* name, LuminosityBlockRange const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getLuminosityBlockRange();
  }

  template <>
  LuminosityBlockRange ParameterSet::getUntrackedParameter<LuminosityBlockRange>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockRange();
  }

  template <>
  std::vector<LuminosityBlockRange> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(
      char const* name, std::vector<LuminosityBlockRange> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->getVLuminosityBlockRange();
  }

  template <>
  std::vector<LuminosityBlockRange> ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(
      char const* name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockRange();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet

  template <>
  ParameterSet ParameterSet::getUntrackedParameter<ParameterSet>(char const* name,
                                                                 ParameterSet const& defaultValue) const {
    return getUntrackedParameterSet(name, defaultValue);
  }

  template <>
  VParameterSet ParameterSet::getUntrackedParameter<VParameterSet>(char const* name,
                                                                   VParameterSet const& defaultValue) const {
    return getUntrackedParameterSetVector(name, defaultValue);
  }

  template <>
  ParameterSet ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name,
                                                                 ParameterSet const& defaultValue) const {
    return getUntrackedParameterSet(name, defaultValue);
  }

  template <>
  VParameterSet ParameterSet::getUntrackedParameter<VParameterSet>(std::string const& name,
                                                                   VParameterSet const& defaultValue) const {
    return getUntrackedParameterSetVector(name, defaultValue);
  }

  template <>
  ParameterSet ParameterSet::getUntrackedParameter<ParameterSet>(char const* name) const {
    return getUntrackedParameterSet(name);
  }

  template <>
  VParameterSet ParameterSet::getUntrackedParameter<VParameterSet>(char const* name) const {
    return getUntrackedParameterSetVector(name);
  }

  template <>
  ParameterSet ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name) const {
    return getUntrackedParameterSet(name);
  }

  template <>
  VParameterSet ParameterSet::getUntrackedParameter<VParameterSet>(std::string const& name) const {
    return getUntrackedParameterSetVector(name);
  }

  //----------------------------------------------------------------------------------
  // specializations for addParameter and addUntrackedParameter

  template <>
  void ParameterSet::addParameter<ParameterSet>(std::string const& name, ParameterSet const& value) {
    invalidateRegistration(name);
    insertParameterSet(true, name, ParameterSetEntry(value, true));
  }

  template <>
  void ParameterSet::addParameter<VParameterSet>(std::string const& name, VParameterSet const& value) {
    invalidateRegistration(name);
    insertVParameterSet(true, name, VParameterSetEntry(value, true));
  }

  template <>
  void ParameterSet::addParameter<ParameterSet>(char const* name, ParameterSet const& value) {
    invalidateRegistration(name);
    insertParameterSet(true, name, ParameterSetEntry(value, true));
  }

  template <>
  void ParameterSet::addParameter<VParameterSet>(char const* name, VParameterSet const& value) {
    invalidateRegistration(name);
    insertVParameterSet(true, name, VParameterSetEntry(value, true));
  }

  template <>
  void ParameterSet::addUntrackedParameter<ParameterSet>(std::string const& name, ParameterSet const& value) {
    insertParameterSet(true, name, ParameterSetEntry(value, false));
  }

  template <>
  void ParameterSet::addUntrackedParameter<VParameterSet>(std::string const& name, VParameterSet const& value) {
    insertVParameterSet(true, name, VParameterSetEntry(value, false));
  }

  template <>
  void ParameterSet::addUntrackedParameter<ParameterSet>(char const* name, ParameterSet const& value) {
    insertParameterSet(true, name, ParameterSetEntry(value, false));
  }

  template <>
  void ParameterSet::addUntrackedParameter<VParameterSet>(char const* name, VParameterSet const& value) {
    insertVParameterSet(true, name, VParameterSetEntry(value, false));
  }

  //----------------------------------------------------------------------------------
  // specializations for getParameterNamesForType

  template <>
  std::vector<std::string> ParameterSet::getParameterNamesForType<ParameterSet>(bool trackiness) const {
    std::vector<std::string> output;
    getParameterSetNames(output, trackiness);
    return output;
  }

  template <>
  std::vector<std::string> ParameterSet::getParameterNamesForType<VParameterSet>(bool trackiness) const {
    std::vector<std::string> output;
    getParameterSetVectorNames(output, trackiness);
    return output;
  }

  ParameterSet const& ParameterSet::getParameterSet(std::string const& name) const {
    return retrieveParameterSet(name).pset();
  }

  ParameterSet const& ParameterSet::getParameterSet(char const* name) const {
    return retrieveParameterSet(name).pset();
  }

  ParameterSet ParameterSet::getUntrackedParameterSet(std::string const& name, ParameterSet const& defaultValue) const {
    return getUntrackedParameterSet(name.c_str(), defaultValue);
  }

  ParameterSet ParameterSet::getUntrackedParameterSet(char const* name, ParameterSet const& defaultValue) const {
    ParameterSetEntry const* entryPtr = retrieveUntrackedParameterSet(name);
    if (entryPtr == nullptr) {
      return defaultValue;
    }
    return entryPtr->pset();
  }

  ParameterSet const& ParameterSet::getUntrackedParameterSet(std::string const& name) const {
    return getUntrackedParameterSet(name.c_str());
  }

  ParameterSet const& ParameterSet::getUntrackedParameterSet(char const* name) const {
    ParameterSetEntry const* result = retrieveUntrackedParameterSet(name);
    if (result == nullptr)
      throw Exception(errors::Configuration, "MissingParameter:")
          << "The required ParameterSet '" << name << "' was not specified.\n";
    return result->pset();
  }

  VParameterSet const& ParameterSet::getParameterSetVector(std::string const& name) const {
    return retrieveVParameterSet(name).vpset();
  }

  VParameterSet const& ParameterSet::getParameterSetVector(char const* name) const {
    return retrieveVParameterSet(name).vpset();
  }

  VParameterSet ParameterSet::getUntrackedParameterSetVector(std::string const& name,
                                                             VParameterSet const& defaultValue) const {
    return getUntrackedParameterSetVector(name.c_str(), defaultValue);
  }

  VParameterSet ParameterSet::getUntrackedParameterSetVector(char const* name,
                                                             VParameterSet const& defaultValue) const {
    VParameterSetEntry const* entryPtr = retrieveUntrackedVParameterSet(name);
    return entryPtr == nullptr ? defaultValue : entryPtr->vpset();
  }

  VParameterSet const& ParameterSet::getUntrackedParameterSetVector(std::string const& name) const {
    return getUntrackedParameterSetVector(name.c_str());
  }

  VParameterSet const& ParameterSet::getUntrackedParameterSetVector(char const* name) const {
    VParameterSetEntry const* result = retrieveUntrackedVParameterSet(name);
    if (result == nullptr)
      throw Exception(errors::Configuration, "MissingParameter:")
          << "The required ParameterSetVector '" << name << "' was not specified.\n";
    return result->vpset();
  }
}  // namespace edm
