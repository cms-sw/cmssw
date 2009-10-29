#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <sstream>
#include <iostream>
namespace edm {

  ParameterSetEntry::ParameterSetEntry()
  : isTracked_(false),
    thePSet_(0),
    theID_()
  {
  }

  ParameterSetEntry::ParameterSetEntry(ParameterSet const& pset, bool isTracked)
  : isTracked_(isTracked),
    thePSet_(new ParameterSet(pset)),
    theID_()
  {
    if (pset.isRegistered()) {
      theID_ = pset.id();
    }
  }

  ParameterSetEntry::ParameterSetEntry(ParameterSetID const& id, bool isTracked)
  : isTracked_(isTracked),
    thePSet_(),
    theID_(id)
  {
  }

  ParameterSetEntry::ParameterSetEntry(std::string const& rep)
  : isTracked_(rep[0] == '+'),
    thePSet_(),
    theID_()
  {
    assert(rep[0] == '+' || rep[0] == '-');
    assert(rep[2] == '(');
    assert(rep[rep.size()-1] == ')');
    ParameterSetID newID(std::string(rep.begin()+3, rep.end()-1) );
    theID_.swap(newID);
  }
    
  ParameterSetEntry::~ParameterSetEntry() {}

  void
  ParameterSetEntry::toString(std::string& result) const {
    result += isTracked() ? "+Q(" : "-Q(";
    if (!theID_.isValid()) {
      throw edm::Exception(edm::errors::LogicError)
        << "ParameterSet::toString() called prematurely\n"
        << "before ParameterSet::registerIt() has been called\n"
        << "for all nested parameter sets\n";
    }
    theID_.toString(result);
    result += ')';
  }

  std::string
  ParameterSetEntry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  ParameterSet const& ParameterSetEntry::pset() const {
    if(!thePSet_) {
      // get it from the registry, and save it here
      thePSet_ = value_ptr<ParameterSet>(new ParameterSet(getParameterSet(theID_)));
    }
    return *thePSet_;
  }

  ParameterSet& ParameterSetEntry::pset() {
    if(!thePSet_) {
      // get it from the registry, and save it here
      thePSet_ = value_ptr<ParameterSet>(new ParameterSet(getParameterSet(theID_)));
    }
    return *thePSet_;
  }

  void ParameterSetEntry::updateID() const {
    assert(pset().isRegistered());
    theID_ = pset().id();
  }

  std::ostream & operator<<(std::ostream & os, ParameterSetEntry const& psetEntry) {
    os << "cms.";
    if(!psetEntry.isTracked()) os << "untracked.";
    os << "PSet(" << psetEntry.pset() << ")";
    return os;
  }
}


