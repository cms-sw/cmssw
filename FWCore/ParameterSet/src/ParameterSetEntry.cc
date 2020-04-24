#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Digest.h"

#include <cassert>
#include <sstream>
#include <iostream>
namespace edm {

  ParameterSetEntry::ParameterSetEntry()
  : isTracked_(false),
    thePSet_(nullptr),
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

  void
  ParameterSetEntry::toDigest(cms::Digest &digest) const {
    digest.append(isTracked() ? "+Q(" : "-Q(", 3);
    if (!theID_.isValid()) {
      throw edm::Exception(edm::errors::LogicError)
        << "ParameterSet::toString() called prematurely\n"
        << "before ParameterSet::registerIt() has been called\n"
        << "for all nested parameter sets\n";
    }
    theID_.toDigest(digest);
    digest.append(")", 1);
  }

  std::string
  ParameterSetEntry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  ParameterSet const& ParameterSetEntry::pset() const {
    fillPSet();
    return *thePSet_;
  }

  ParameterSet& ParameterSetEntry::psetForUpdate() {
    fillPSet();
    return *thePSet_;
  }

  void ParameterSetEntry::fillPSet() const {
    if(nullptr == thePSet_.load()) {
      auto tmp = std::make_unique<ParameterSet>(getParameterSet(theID_));
      ParameterSet* expected = nullptr;
      if(thePSet_.compare_exchange_strong(expected, tmp.get())) {
        // thePSet_ was equal to nullptr and now is equal to tmp.get()
        tmp.release();
      }
    }
  }

  void ParameterSetEntry::updateID() {
    assert(pset().isRegistered());
    theID_ = pset().id();
  }

  std::string ParameterSetEntry::dump(unsigned int indent) const {
    std::ostringstream os;
    const char* trackiness = (isTracked()?"tracked":"untracked");
    os << "PSet "<<trackiness<<" = (" << pset().dump(indent) << ")";
    return os.str();
  }

  std::ostream & operator<<(std::ostream & os, ParameterSetEntry const& psetEntry) {
    os << psetEntry.dump();
    return os;
  }
}


