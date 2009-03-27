#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/split.h"
namespace edm {

  VParameterSetEntry::VParameterSetEntry()
  : tracked(false),
    theVPSet(),
    theIDs()
  {
  }

  VParameterSetEntry::VParameterSetEntry(std::vector<ParameterSet> const& vpset, bool isTracked)
  : tracked(isTracked),
    theVPSet(new std::vector<ParameterSet>),
    theIDs()
  {
    for (std::vector<ParameterSet>::const_iterator i = vpset.begin(), e = vpset.end(); i != e; ++i) {
      theVPSet->push_back(*i);
    }
  }

  VParameterSetEntry::VParameterSetEntry(std::string const& rep)
  : tracked(rep[0] == '+'),
    theVPSet(),
    theIDs(new std::vector<ParameterSetID>)
  {
    assert(rep[0] == '+' || rep[0] == '-');
    std::vector<std::string> temp;
    // need a substring that starts at the '{'
    std::string bracketedRepr(rep.begin()+2, rep.end());
    split(std::back_inserter(temp), bracketedRepr, '{', ',', '}');
    theIDs->reserve(temp.size());
    for (std::vector<std::string>::const_iterator i = temp.begin(), e = temp.end(); i != e; ++i) {
      theIDs->push_back(ParameterSetID(*i));
    }
  }

  VParameterSetEntry::~VParameterSetEntry() {}

  void
  VParameterSetEntry::toString(std::string& result) const {
    assert(theIDs);
    result += tracked ? "+q" : "-q";
    result += '{';
    std::string start;
    std::string const between(",");
    for (std::vector<ParameterSetID>::const_iterator i = theIDs->begin(), e = theIDs->end(); i != e; ++i) {
      result += start;
      i->toString(result);
      start = between;
    }
    result += '}';
  }

  std::string VParameterSetEntry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  std::vector<ParameterSet> const& VParameterSetEntry::vpset() const {
    if (!theVPSet) {
      assert(theIDs);
      theVPSet = value_ptr<std::vector<ParameterSet> >(new std::vector<ParameterSet>);
      theVPSet->reserve(theIDs->size());
      for (std::vector<ParameterSetID>::const_iterator i = theIDs->begin(), e = theIDs->end(); i != e; ++i) {
        theVPSet->push_back(getParameterSet(*i));
      }
    }
    return *theVPSet;
  }

  ParameterSet & VParameterSetEntry::psetInVector(int i) {
    assert(theVPSet);
    return theVPSet->at(i);
  }

  void VParameterSetEntry::registerPsetsAndUpdateIDs() {
    vpset();
    theIDs = value_ptr<std::vector<ParameterSetID> >(new std::vector<ParameterSetID>);
    theIDs->resize(theVPSet->size());
    for (std::vector<ParameterSet>::iterator i = theVPSet->begin(), e = theVPSet->end(); i != e; ++i) {
      if (!i->isRegistered()) {
        i->registerIt();
      }
      theIDs->at(i - theVPSet->begin()) = i->id();
    }
  }

  std::ostream & operator<<(std::ostream & os, VParameterSetEntry const& vpsetEntry) {
    std::vector<ParameterSet> const& vps = vpsetEntry.vpset();
    os << "{" << std::endl;
    std::string start;
    std::string const between(",\n");
    for(std::vector<ParameterSet>::const_iterator i = vps.begin(), e = vps.end(); i != e; ++i) {
      os << start << *i;
      start = between;
    }
    if (!vps.empty()) {
      os << std::endl;
    }
    os << "}";
    return os;
  }
}
