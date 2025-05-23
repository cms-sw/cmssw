
#include "FWCore/ParameterSet/src/FillDescriptionFromPSet.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Entry.h"

#include <string>
#include <map>
#include <vector>

namespace edm {
  class EventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class ESInputTag;
  class FileInPath;
}  // namespace edm

typedef void (*FillDescriptionFromParameter)(edm::ParameterSet const&,
                                             std::string const&,
                                             bool,
                                             edm::ParameterSetDescription&);

namespace {

  template <typename T>
  void fillDescriptionFromParameter(edm::ParameterSet const& pset,
                                    std::string const& name,
                                    bool isTracked,
                                    edm::ParameterSetDescription& desc) {
    if (isTracked) {
      desc.add<T>(name, pset.getParameter<T>(name));
    } else {
      desc.addUntracked<T>(name, pset.getUntrackedParameter<T>(name));
    }
  }

  std::map<edm::ParameterTypes, FillDescriptionFromParameter> initMap() {
    std::map<edm::ParameterTypes, FillDescriptionFromParameter> findTheRightFunction;
    findTheRightFunction[edm::k_int32] = &fillDescriptionFromParameter<int>;
    findTheRightFunction[edm::k_vint32] = &fillDescriptionFromParameter<std::vector<int>>;
    findTheRightFunction[edm::k_uint32] = &fillDescriptionFromParameter<unsigned>;
    findTheRightFunction[edm::k_vuint32] = &fillDescriptionFromParameter<std::vector<unsigned>>;
    findTheRightFunction[edm::k_int64] = &fillDescriptionFromParameter<long long>;
    findTheRightFunction[edm::k_vint64] = &fillDescriptionFromParameter<std::vector<long long>>;
    findTheRightFunction[edm::k_uint64] = &fillDescriptionFromParameter<unsigned long long>;
    findTheRightFunction[edm::k_vuint64] = &fillDescriptionFromParameter<std::vector<unsigned long long>>;
    findTheRightFunction[edm::k_double] = &fillDescriptionFromParameter<double>;
    findTheRightFunction[edm::k_vdouble] = &fillDescriptionFromParameter<std::vector<double>>;
    findTheRightFunction[edm::k_bool] = &fillDescriptionFromParameter<bool>;
    findTheRightFunction[edm::k_stringRaw] = &fillDescriptionFromParameter<std::string>;
    findTheRightFunction[edm::k_vstringRaw] = &fillDescriptionFromParameter<std::vector<std::string>>;
    findTheRightFunction[edm::k_EventID] = &fillDescriptionFromParameter<edm::EventID>;
    findTheRightFunction[edm::k_VEventID] = &fillDescriptionFromParameter<std::vector<edm::EventID>>;
    findTheRightFunction[edm::k_LuminosityBlockID] = &fillDescriptionFromParameter<edm::LuminosityBlockID>;
    findTheRightFunction[edm::k_VLuminosityBlockID] =
        &fillDescriptionFromParameter<std::vector<edm::LuminosityBlockID>>;
    findTheRightFunction[edm::k_InputTag] = &fillDescriptionFromParameter<edm::InputTag>;
    findTheRightFunction[edm::k_VInputTag] = &fillDescriptionFromParameter<std::vector<edm::InputTag>>;
    findTheRightFunction[edm::k_ESInputTag] = &fillDescriptionFromParameter<edm::ESInputTag>;
    findTheRightFunction[edm::k_VESInputTag] = &fillDescriptionFromParameter<std::vector<edm::ESInputTag>>;
    findTheRightFunction[edm::k_FileInPath] = &fillDescriptionFromParameter<edm::FileInPath>;
    findTheRightFunction[edm::k_LuminosityBlockRange] = &fillDescriptionFromParameter<edm::LuminosityBlockRange>;
    findTheRightFunction[edm::k_VLuminosityBlockRange] =
        &fillDescriptionFromParameter<std::vector<edm::LuminosityBlockRange>>;
    findTheRightFunction[edm::k_EventRange] = &fillDescriptionFromParameter<edm::EventRange>;
    findTheRightFunction[edm::k_VEventRange] = &fillDescriptionFromParameter<std::vector<edm::EventRange>>;
    return findTheRightFunction;
  }

  std::map<edm::ParameterTypes, FillDescriptionFromParameter> const s_findTheRightFunction = initMap();

  std::map<edm::ParameterTypes, FillDescriptionFromParameter> const& findTheRightFunction() {
    return s_findTheRightFunction;
  }
}  // namespace

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

  void fillDescriptionFromPSet(ParameterSet const& pset, ParameterSetDescription& desc) {
    ParameterSet::table const& entries = pset.tbl();
    for (ParameterSet::table::const_iterator entry = entries.begin(), endEntries = entries.end(); entry != endEntries;
         ++entry) {
      std::map<edm::ParameterTypes, FillDescriptionFromParameter>::const_iterator iter =
          findTheRightFunction().find(static_cast<edm::ParameterTypes>(entry->second.typeCode()));
      if (iter != findTheRightFunction().end()) {
        iter->second(pset, entry->first, entry->second.isTracked(), desc);
      }
    }

    ParameterSet::psettable const& pset_entries = pset.psetTable();
    for (ParameterSet::psettable::const_iterator pset_entry = pset_entries.begin(), endEntries = pset_entries.end();
         pset_entry != endEntries;
         ++pset_entry) {
      edm::ParameterSet nestedPset;
      if (pset_entry->second.isTracked()) {
        nestedPset = pset.getParameterSet(pset_entry->first);
      } else {
        nestedPset = pset.getUntrackedParameterSet(pset_entry->first);
      }
      ParameterSetDescription nestedDescription;
      fillDescriptionFromPSet(nestedPset, nestedDescription);
      if (pset_entry->second.isTracked()) {
        desc.add<edm::ParameterSetDescription>(pset_entry->first, nestedDescription);
      } else {
        desc.addUntracked<edm::ParameterSetDescription>(pset_entry->first, nestedDescription);
      }
    }

    ParameterSet::vpsettable const& vpset_entries = pset.vpsetTable();
    for (ParameterSet::vpsettable::const_iterator vpset_entry = vpset_entries.begin(), endEntries = vpset_entries.end();
         vpset_entry != endEntries;
         ++vpset_entry) {
      std::vector<edm::ParameterSet> nestedVPset;
      if (vpset_entry->second.isTracked()) {
        nestedVPset = pset.getParameterSetVector(vpset_entry->first);
      } else {
        nestedVPset = pset.getUntrackedParameterSetVector(vpset_entry->first);
      }
      ParameterSetDescription emptyDescription;

      auto pd = std::make_unique<ParameterDescription<std::vector<ParameterSet>>>(
          vpset_entry->first, emptyDescription, vpset_entry->second.isTracked(), nestedVPset);

      pd->setPartOfDefaultOfVPSet(true);
      std::unique_ptr<ParameterDescriptionNode> node(std::move(pd));
      desc.addNode(std::move(node));
    }
  }
}  // namespace edm
