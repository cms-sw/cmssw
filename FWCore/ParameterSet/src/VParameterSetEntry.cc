#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/split.h"
#include "FWCore/Utilities/interface/Digest.h"

#include <cassert>
#include <ostream>
#include <sstream>

namespace edm {

  VParameterSetEntry::VParameterSetEntry() :
      tracked_(false),
      theVPSet_(),
      theIDs_() {
  }

  VParameterSetEntry::VParameterSetEntry(std::vector<ParameterSet> const& vpset, bool isTracked) :
      tracked_(isTracked),
      theVPSet_(new std::vector<ParameterSet>),
      theIDs_() {
    for (std::vector<ParameterSet>::const_iterator i = vpset.begin(), e = vpset.end(); i != e; ++i) {
      theVPSet_->push_back(*i);
    }
  }

  VParameterSetEntry::VParameterSetEntry(std::string const& rep) :
      tracked_(rep[0] == '+'),
      theVPSet_(),
      theIDs_(new std::vector<ParameterSetID>) {
    assert(rep[0] == '+' || rep[0] == '-');
    std::vector<std::string> temp;
    // need a substring that starts at the '{'
    std::string bracketedRepr(rep.begin()+2, rep.end());
    split(std::back_inserter(temp), bracketedRepr, '{', ',', '}');
    theIDs_->reserve(temp.size());
    for (std::vector<std::string>::const_iterator i = temp.begin(), e = temp.end(); i != e; ++i) {
      theIDs_->push_back(ParameterSetID(*i));
    }
  }

  void
  VParameterSetEntry::toString(std::string& result) const {
    assert(theIDs_);
    result += tracked_ ? "+q" : "-q";
    result += '{';
    std::string start;
    std::string const between(",");
    for (std::vector<ParameterSetID>::const_iterator i = theIDs_->begin(), e = theIDs_->end(); i != e; ++i) {
      result += start;
      i->toString(result);
      start = between;
    }
    result += '}';
  }

  void
  VParameterSetEntry::toDigest(cms::Digest &digest) const {
    assert(theIDs_);
    digest.append(tracked_ ? "+q{" : "-q{", 3);
    bool started = false;
    for (std::vector<ParameterSetID>::const_iterator i = theIDs_->begin(), e = theIDs_->end(); i != e; ++i) {
      if (started)
        digest.append(",", 1);
      i->toDigest(digest);
      started = true;
    }
    digest.append("}",1);
  }

  std::string VParameterSetEntry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  std::vector<ParameterSet> const& VParameterSetEntry::vpset() const {
    fillVPSet();
    return *theVPSet_;
  }

  // NOTE: This function, and other non-const functions of this class
  // that expose internals, may be used in a way that causes the cached
  // "theVPSet_" and "theIDs_" to be inconsistent.
  // THIS PROBLEM NEEDS TO BE ADDRESSED
  std::vector<ParameterSet>& VParameterSetEntry::vpsetForUpdate() {
    fillVPSet();
    return *theVPSet_;
  }

  void VParameterSetEntry::fillVPSet() const {
    if(nullptr == theVPSet_.load()) {
      auto tmp = std::make_unique<std::vector<ParameterSet>>();
      tmp->reserve(theIDs_->size());
      for (auto const& theID : *theIDs_) {
        tmp->push_back(getParameterSet(theID));
      }
      VParameterSet* expected = nullptr;
      if(theVPSet_.compare_exchange_strong(expected, tmp.get())) {
        // theVPSet_ was equal to nullptr and now is equal to tmp.get()
        tmp.release();
      }
    }
  }

  // NOTE: This function, and other non-const functions of this class
  // that expose internals, may be used in a way that causes the cached
  // "theVPSet_" and "theIDs_" to be inconsistent.
  // THIS PROBLEM NEEDS TO BE ADDRESSED
  ParameterSet& VParameterSetEntry::psetInVector(int i) {
    assert(theVPSet_);
    return theVPSet_->at(i);
  }

  std::vector<ParameterSet>::size_type VParameterSetEntry::size() const {
    return vpset().size();
  }

  void VParameterSetEntry::registerPsetsAndUpdateIDs() {
    fillVPSet();
    theIDs_ = value_ptr<std::vector<ParameterSetID> >(new std::vector<ParameterSetID>);
    theIDs_->resize(theVPSet_->size());
    for (std::vector<ParameterSet>::iterator i = theVPSet_->begin(), e = theVPSet_->end(); i != e; ++i) {
      if (!i->isRegistered()) {
        i->registerIt();
      }
      theIDs_->at(i - theVPSet_->begin()) = i->id();
    }
  }

  std::string VParameterSetEntry::dump(unsigned int indent)  const {
    std::string indentation(indent, ' ');
    std::ostringstream os;
    std::vector<ParameterSet> const& vps = vpset();
    os << "VPSet "<<(isTracked()?"tracked":"untracked")<<" = ({" << std::endl;
    std::string start;
    std::string const between(",\n");
    for(std::vector<ParameterSet>::const_iterator i = vps.begin(), e = vps.end(); i != e; ++i) {
      os << start << indentation << i->dump(indent);
      start = between;
    }
    if (!vps.empty()) {
      os << std::endl;
    }
    os << indentation << "})";
    return os.str();
  }

  std::ostream& operator<<(std::ostream& os, VParameterSetEntry const& vpsetEntry) {
    os << vpsetEntry.dump();
    return os;
  }
}
