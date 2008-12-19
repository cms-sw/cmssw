#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include <sstream>
#include <iostream>
namespace edm {

  ParameterSetEntry::ParameterSetEntry()
  : tracked(false),
    thePSet(0),
    theID()
  {
  }

  ParameterSetEntry::ParameterSetEntry(ParameterSet const& pset, bool isTracked)
  : tracked(isTracked),
    thePSet(new ParameterSet(pset)),
    theID()
  {
  }

  ParameterSetEntry::ParameterSetEntry(std::string const& rep)
  : tracked(rep[0] == '+'),
    thePSet(),
    theID()
  {
    assert(rep[0] == '+' || rep[0] == '-');
    ParameterSetID newID(std::string(rep.begin()+2, rep.end()) );
    theID.swap(newID);
  }
    
  ParameterSetEntry::~ParameterSetEntry() {}

  void ParameterSetEntry::toString(std::string& result) const {
    result += tracked ? "+Q" : "-Q";
    theID.toString(result);
  }
  
  int ParameterSetEntry::sizeOfString() const {
    std::string str;
    toString(str);
    return str.size();
  }

  ParameterSet & ParameterSetEntry::pset() {
    if(!thePSet) {
      // get it from the registry, and save it here
      thePSet = value_ptr<ParameterSet>(new ParameterSet( getParameterSet(theID) ));
    }
    return *thePSet;
  }

  ParameterSet const& ParameterSetEntry::pset() const {
    if(!thePSet) {
      // get it from the registry, and save it here
      thePSet = value_ptr<ParameterSet>(new ParameterSet( getParameterSet(theID) ));
    }
    return *thePSet;

  }

  void ParameterSetEntry::updateID() const {
    theID = pset().id();
    pset().setID(theID);
  }

  std::ostream & operator<<(std::ostream & os, ParameterSetEntry const& psetEntry) {
    os << "cms.";
    if(!psetEntry.isTracked()) os << "untracked.";
    os << "PSet(" << psetEntry.pset() << ")";
    return os;
  }
}


