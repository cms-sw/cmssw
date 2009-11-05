
#include "FWCore/ParameterSet/interface/FillDescriptionFromPSet.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Entry.h"

#include <boost/cstdint.hpp>

#include <string>
#include <map>
#include <vector>

namespace edm {
  class MinimalEventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class FileInPath;
}

typedef void (*FillDescriptionFromParameter)(edm::ParameterSet const&, std::string const&, bool, edm::ParameterSetDescription &);

static std::map<edm::ParameterTypes, FillDescriptionFromParameter> s_findTheRightFunction;

namespace {

  template <typename T>
  void fillDescriptionFromParameter(edm::ParameterSet const& pset,
                                    std::string const& name,
                                    bool isTracked,
                                    edm::ParameterSetDescription & desc) {
    if (isTracked) {
      desc.add<T>(name, pset.getParameter<T>(name));
    }
    else {
      desc.addUntracked<T>(name, pset.getUntrackedParameter<T>(name));
    }
  }

  void initMap() {
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('I')] = &fillDescriptionFromParameter<int>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('i')] = &fillDescriptionFromParameter<std::vector<int> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('U')] = &fillDescriptionFromParameter<unsigned>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('u')] = &fillDescriptionFromParameter<std::vector<unsigned> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('L')] = &fillDescriptionFromParameter<long long>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('l')] = &fillDescriptionFromParameter<std::vector<long long> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('X')] = &fillDescriptionFromParameter<unsigned long long>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('x')] = &fillDescriptionFromParameter<std::vector<unsigned long long> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('D')] = &fillDescriptionFromParameter<double>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('d')] = &fillDescriptionFromParameter<std::vector<double> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('B')] = &fillDescriptionFromParameter<bool>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('S')] = &fillDescriptionFromParameter<std::string>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('s')] = &fillDescriptionFromParameter<std::vector<std::string> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('E')] = &fillDescriptionFromParameter<edm::MinimalEventID>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('e')] = &fillDescriptionFromParameter<std::vector<edm::MinimalEventID> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('M')] = &fillDescriptionFromParameter<edm::LuminosityBlockID>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('m')] = &fillDescriptionFromParameter<std::vector<edm::LuminosityBlockID> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('t')] = &fillDescriptionFromParameter<edm::InputTag>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('v')] = &fillDescriptionFromParameter<std::vector<edm::InputTag> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('F')] = &fillDescriptionFromParameter<edm::FileInPath>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('A')] = &fillDescriptionFromParameter<edm::LuminosityBlockRange>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('a')] = &fillDescriptionFromParameter<std::vector<edm::LuminosityBlockRange> >;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('R')] = &fillDescriptionFromParameter<edm::EventRange>;
    s_findTheRightFunction[static_cast<edm::ParameterTypes>('r')] = &fillDescriptionFromParameter<std::vector<edm::EventRange> >;
  }

  std::map<edm::ParameterTypes, FillDescriptionFromParameter>& findTheRightFunction() {
    if (s_findTheRightFunction.empty()) initMap();
    return s_findTheRightFunction;
  }
}

namespace edm {

  // Note that the description this fills is used for purposes
  // of printing documentation from edmPluginHelp and writing
  // cfi files.  In general, it will not be useful for validation
  // purposes.  First of all, if the ParameterSet contains a
  // vector of ParameterSets, then the description of that vector
  // of ParameterSets will have an empty ParameterSetDescription
  // (so if you try to validate with such a description, it will
  // always fail).  Also, the ParameterSet has no concept of "optional"
  // or the logical relationships between parameters in the
  // description (like "and", "xor", "switches", ...), so there is
  // no way a description generated from a ParameterSet can properly
  // express those concepts.

  void fillDescriptionFromPSet(ParameterSet const& pset,
                               ParameterSetDescription & desc) {
    ParameterSet::table const& entries = pset.tbl();
    for (ParameterSet::table::const_iterator entry = entries.begin(),
	   endEntries = entries.end();
         entry != endEntries;
         ++entry) {
      std::map<edm::ParameterTypes, FillDescriptionFromParameter>::iterator iter =
        findTheRightFunction().find(static_cast<edm::ParameterTypes>(entry->second.typeCode()));
      if (iter != findTheRightFunction().end()) {
        iter->second(pset, entry->first, entry->second.isTracked(), desc);
      }
    }

    ParameterSet::psettable const& pset_entries = pset.psetTable();
    for (ParameterSet::psettable::const_iterator pset_entry = pset_entries.begin(),
	   endEntries = pset_entries.end();
         pset_entry != endEntries;
         ++pset_entry) {
      edm::ParameterSet nestedPset;
      if (pset_entry->second.isTracked()) {
        nestedPset = pset.getParameterSet(pset_entry->first);
      }
      else {
        nestedPset = pset.getUntrackedParameterSet(pset_entry->first);
      }
      ParameterSetDescription nestedDescription;
      fillDescriptionFromPSet(nestedPset, nestedDescription);
      if (pset_entry->second.isTracked()) {
        desc.add<edm::ParameterSetDescription>(pset_entry->first, nestedDescription);
      }
      else {
        desc.addUntracked<edm::ParameterSetDescription>(pset_entry->first, nestedDescription);
      }
    }

    ParameterSet::vpsettable const& vpset_entries = pset.vpsetTable();
    for (ParameterSet::vpsettable::const_iterator vpset_entry = vpset_entries.begin(),
	   endEntries = vpset_entries.end();
         vpset_entry != endEntries;
         ++vpset_entry) {
      std::vector<edm::ParameterSet> nestedVPset;
      if (vpset_entry->second.isTracked()) {
        nestedVPset = pset.getParameterSetVector(vpset_entry->first);
      }
      else {
        nestedVPset = pset.getUntrackedParameterSetVector(vpset_entry->first);
      }
      ParameterSetDescription emptyDescription;

      std::auto_ptr<ParameterDescription<std::vector<ParameterSet> > >
        pd(new ParameterDescription<std::vector<ParameterSet> >(vpset_entry->first, emptyDescription, vpset_entry->second.isTracked(), nestedVPset));

      pd->setPartOfDefaultOfVPSet(true);
      std::auto_ptr<ParameterDescriptionNode> node(pd);
      desc.addNode(node);
    }
  }
}
