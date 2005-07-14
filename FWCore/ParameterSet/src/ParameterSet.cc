// ----------------------------------------------------------------------
// $Id: ParameterSet.cc,v 1.4 2005/06/23 19:57:23 wmtan Exp $
//
// definition of ParameterSet's function members
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/split.h"
#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <utility>

// ----------------------------------------------------------------------
// class invariant checker
// ----------------------------------------------------------------------

namespace edm {

  void
  ParameterSet::validate() const {
  }  // ParameterSet::validate()
  
  
  // ----------------------------------------------------------------------
  // constructors
  // ----------------------------------------------------------------------
  
  // ----------------------------------------------------------------------
  // coded string
  
  ParameterSet::ParameterSet(std::string const& code) : tbl() {
    if(! fromString(code))
      throw edm::Exception(errors::Configuration,"InvalidInput")
	<< "The encoded configuration string "
	<< "passed to a ParameterSet during construction is invalid:\n"
	<< code;

    validate();
  }
  
  // ----------------------------------------------------------------------
  // Entry-handling
  // ----------------------------------------------------------------------
  
  Entry const&
  ParameterSet::retrieve(std::string const& name) const {
    table::const_iterator  it = tbl.find(name);
    if(it == tbl.end()) {
      it = tbl.find("label");
      if(it == tbl.end())
        throw edm::Exception(errors::Configuration,"InvalidName")
	  << "The name '" << name 
	  << "' is not known in an anonymous ParameterSet";
      else
        throw edm::Exception(errors::Configuration,"InvalidName")
	  << "The name '" << name
	  << "' is not known in ParameterSet '"
	  << it->second.getString() << "'";
    }
    return it->second;
  }  // retrieve()
  
  Entry const* const
  ParameterSet::retrieveUntracked(std::string const& name) const {
    table::const_iterator  it = tbl.find(name);
    if(it == tbl.end())  {
      return 0;
    }
    return &it->second;
  }  // retrieve()
  
  // ----------------------------------------------------------------------
  
  void
  ParameterSet::insert(bool okay_to_replace, std::string const& name, Entry const& value) {
    table::iterator  it = tbl.find(name);
  
    if(it == tbl.end())  {
      if(! tbl.insert(std::make_pair(name, value)).second)
        throw edm::Exception(errors::Configuration,"InsertFailure")
	  << "cannot insert " << name
	  << " into a ParmeterSet";
    }
    else if(okay_to_replace)  {
      it->second = value;
    }
  }  // insert()
  
  // ----------------------------------------------------------------------
  // copy without overwriting
  // ----------------------------------------------------------------------
  
  void
  ParameterSet::augment(ParameterSet const& from) {
    if(& from == this)
      return;
  
    for(table::const_iterator b = from.tbl.begin(), e = from.tbl.end(); b != e; ++b) {
      this->insert(false, b->first, b->second);
    }
  }  // augment()
  
  // ----------------------------------------------------------------------
  // coding
  // ----------------------------------------------------------------------
  
  std::string
  ParameterSet::toString() const {
    std::string rep;
    for(table::const_iterator b = tbl.begin(), e = tbl.end(); b != e; ++b) {
      if(b != tbl.begin())
        rep += ';';
      rep += (b->first + '=' + b->second.toString());
    }
  
    return '<' + rep + '>';
  }  // to_string()
  
  // ----------------------------------------------------------------------
  
  std::string
  ParameterSet::toStringOfTracked() const {
    std::string  rep = "<";
    bool need_sep = false;
    for(table::const_iterator b = tbl.begin(), e = tbl.end(); b != e; ++b) {
      if(b->second.isTracked())  {
        if(need_sep)
          rep += ';';
        rep += (b->first + '=' + b->second.toString());
        need_sep = true;
      }
    }
  
    return rep + '>';
  }  // to_string()
  
  // ----------------------------------------------------------------------
  
  bool
  ParameterSet::fromString(std::string const& from) {
    std::vector<std::string> temp;
    if(! split(std::back_inserter(temp), from, '<', ';', '>'))
      return false;
  
    tbl.clear();  // precaution
    for(std::vector<std::string>::const_iterator b = temp.begin(), e = temp.end(); b != e; ++b) {
      // locate required name/value separator
      std::string::const_iterator  q
        = std::find(b->begin(), b->end(), '=');
      if(q == b->end())
        return false;
  
      // form name unique to this ParameterSet
      std::string  name = std::string(b->begin(), q);
      if(tbl.find(name) != tbl.end())
        return false;
  
      // form value and insert name/value pair
      Entry  value(std::string(q+1, b->end()));
      if(! tbl.insert(std::make_pair(name, value)).second)
        return false;
    }
  
    return true;
  }  // from_string()
  
} // namespace edm
// ----------------------------------------------------------------------
