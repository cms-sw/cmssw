#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/split.h"
namespace edm {

  VParameterSetEntry::VParameterSetEntry()
  : tracked(false),
    theVPSet(),
    thePSetEntries()
  {
  }

  VParameterSetEntry::VParameterSetEntry(std::vector<ParameterSet> const& vpset, bool isTracked)
  : tracked(isTracked),
    theVPSet(),
    thePSetEntries()
  {
    for (std::vector<ParameterSet>::const_iterator i = vpset.begin(), e = vpset.end(); i != e; ++i) {
      thePSetEntries.push_back(ParameterSetEntry(*i, isTracked));
    }
  }

  VParameterSetEntry::VParameterSetEntry(std::string const& rep)
  : tracked(rep[0] == '+'),
    theVPSet(),
    thePSetEntries()
  {
    assert(rep[0] == '+' || rep[0] == '-');
    std::vector<std::string> temp;
    // need a substring that starts at the '{'
    std::string bracketedRepr(rep.begin()+2, rep.end());
    split(std::back_inserter(temp), bracketedRepr, '{', ',', '}');
    for (std::vector<std::string>::const_iterator i = temp.begin(), e = temp.end(); i != e; ++i) {
      thePSetEntries.push_back(ParameterSetEntry(*i));
    }
  }
    
  VParameterSetEntry::~VParameterSetEntry() {}

  void VParameterSetEntry::updateIDs() const {
    for (std::vector<ParameterSetEntry>::const_iterator i = psetEntries().begin(), e = psetEntries().end();
        i != e; ++i) {
      i->updateID();
    }
    // clear theVPSet
    value_ptr<VParameterSet> empty;
    swap(theVPSet, empty);
  }

  void VParameterSetEntry::toString(std::string& result) const {
    result += tracked ? "+q" : "-q";
    result += '{';
    std::string start;
    std::string const between(",");
    for (std::vector<ParameterSetEntry>::const_iterator i = thePSetEntries.begin(), e = thePSetEntries.end(); i != e; ++i) {
      result += start;
      i->toString(result);
      start = between;
    }
    result += '}';
  }
  
  int VParameterSetEntry::sizeOfString() const {
    std::string str;
    toString(str);
    return str.size();
  }

  std::vector<ParameterSet>& VParameterSetEntry::vpset() {
    if (!theVPSet) {
      theVPSet = value_ptr<std::vector<ParameterSet> >(new std::vector<ParameterSet>);
      theVPSet->reserve(thePSetEntries.size());
      for (std::vector<ParameterSetEntry>::const_iterator i = thePSetEntries.begin(), e = thePSetEntries.end(); i != e; ++i) {
        theVPSet->push_back(i->pset());
      }
    }
    return *theVPSet;
  }

  std::vector<ParameterSet> const& VParameterSetEntry::vpset() const {
    if (!theVPSet) {
      theVPSet = value_ptr<std::vector<ParameterSet> >(new std::vector<ParameterSet>);
      theVPSet->reserve(thePSetEntries.size());
      for (std::vector<ParameterSetEntry>::const_iterator i = thePSetEntries.begin(), e = thePSetEntries.end(); i != e; ++i) {
        theVPSet->push_back(i->pset());
      }
    }
    return *theVPSet;
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


