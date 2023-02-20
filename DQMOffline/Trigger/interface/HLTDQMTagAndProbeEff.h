#ifndef DQMOffline_Trigger_HLTTagAndProbeEff_h
#define DQMOffline_Trigger_HLTTagAndProbeEff_h

//***************************************************************************
//
// Description:
//   This class produces histograms which can be used to make tag&probe efficiencies
//   for specified HLT filters
//   These histograms can be binned in various variables
//
//   In a nutshell:
//   1) requires the tag trigger to pass for the event
//   2) creates a collection of tags, with the tags required to pass the a specified
//      filter and selection ID. The selection is in the form of a value map calculated
//      by a previous module. Additional kinematic cuts can be applied (its not limited to
//      kinematic cuts per se but thats the intention)
//   3) likewise creates a collection of probes passing filter, ID and kinematic requirements
//   4) applies selection like mass & opp sign to the t&p pair
//   5) passes the t&p pair to HLTDQMFilterTnPEffHists, with each one corresponding to a
//      specific HLT filter whose efficiency we wish to measure. This class then creates the
//      numerator and denominator histograms for the efficiency calculation done at harvesting
//
//
// Author: Sam Harper (RAL) , 2017
//
//***************************************************************************

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DQMOffline/Trigger/interface/UtilFuncs.h"
#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"
#include "DQMOffline/Trigger/interface/HLTDQMFilterTnPEffHists.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include <vector>
#include <string>

namespace {
  template <typename T>
  edm::Handle<T> getHandle(const edm::Event& event, const edm::EDGetTokenT<T>& token) {
    edm::Handle<T> handle;
    event.getByToken(token, handle);
    return handle;
  }
}  // namespace

template <typename TagType, typename TagCollType, typename ProbeType = TagType, typename ProbeCollType = TagCollType>
class HLTDQMTagAndProbeEff {
public:
  typedef dqm::legacy::DQMStore DQMStore;

  explicit HLTDQMTagAndProbeEff(const edm::ParameterSet& pset, edm::ConsumesCollector&& cc);

  static edm::ParameterSetDescription makePSetDescription();

  void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  void bookHists(DQMStore::IBooker& iBooker);
  void fill(const edm::Event& event, const edm::EventSetup& setup);

private:
  template <typename ObjType, typename ObjCollType>
  std::vector<edm::Ref<ObjCollType> > getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
                                                     const trigger::TriggerEvent& trigEvt,
                                                     const std::vector<std::string>& filterNames,
                                                     const bool orFilters,
                                                     const edm::Handle<edm::ValueMap<bool> >& vidHandle,
                                                     const VarRangeCutColl<ObjType>& rangeCuts);

private:
  edm::EDGetTokenT<TagCollType> tagToken_;
  edm::EDGetTokenT<ProbeCollType> probeToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> trigEvtToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > tagVIDToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > probeVIDToken_;

  std::string hltProcess_;

  std::string tagTrigger_;

  std::vector<std::string> tagFilters_;
  bool tagFiltersORed_;  //true=ORed, false=ANDed
  VarRangeCutColl<TagType> tagRangeCuts_;

  std::vector<std::string> probeFilters_;
  bool probeFiltersORed_;  //true=ORed, false=ANDed
  VarRangeCutColl<ProbeType> probeRangeCuts_;

  float minTagProbeDR2_;
  float minMass_;
  float maxMass_;
  bool requireOpSign_;

  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<HLTDQMFilterTnPEffHists<TagType, ProbeType> > filterHists_;

  GenericTriggerEventFlag sampleTrigRequirements_;
};

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::HLTDQMTagAndProbeEff(
    const edm::ParameterSet& pset, edm::ConsumesCollector&& cc)
    : tagRangeCuts_(pset.getParameter<std::vector<edm::ParameterSet> >("tagRangeCuts")),
      probeRangeCuts_(pset.getParameter<std::vector<edm::ParameterSet> >("probeRangeCuts")),
      sampleTrigRequirements_(pset.getParameter<edm::ParameterSet>("sampleTrigRequirements"), cc) {
  edm::InputTag trigEvtTag = pset.getParameter<edm::InputTag>("trigEvent");

  tagToken_ = cc.consumes<TagCollType>(pset.getParameter<edm::InputTag>("tagColl"));
  probeToken_ = cc.consumes<ProbeCollType>(pset.getParameter<edm::InputTag>("probeColl"));
  trigEvtToken_ = cc.consumes<trigger::TriggerEvent>(trigEvtTag);
  tagVIDToken_ = cc.consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("tagVIDCuts"));
  probeVIDToken_ = cc.consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("probeVIDCuts"));

  hltProcess_ = trigEvtTag.process();

  tagFilters_ = pset.getParameter<std::vector<std::string> >("tagFilters");
  tagFiltersORed_ = pset.getParameter<bool>("tagFiltersORed");
  probeFilters_ = pset.getParameter<std::vector<std::string> >("probeFilters");
  probeFiltersORed_ = pset.getParameter<bool>("tagFiltersORed");

  double minDR = pset.getParameter<double>("minTagProbeDR");
  minTagProbeDR2_ = minDR * minDR;
  minMass_ = pset.getParameter<double>("minMass");
  maxMass_ = pset.getParameter<double>("maxMass");
  requireOpSign_ = pset.getParameter<bool>("requireOpSign");

  histConfigs_ = pset.getParameter<std::vector<edm::ParameterSet> >("histConfigs");
  const auto& filterConfigs = pset.getParameter<std::vector<edm::ParameterSet> >("filterConfigs");

  std::string baseHistName = pset.getParameter<std::string>("baseHistName");

  for (auto& config : filterConfigs) {
    filterHists_.emplace_back(HLTDQMFilterTnPEffHists<TagType, ProbeType>(config, baseHistName, hltProcess_));
  }
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
edm::ParameterSetDescription HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::makePSetDescription() {
  edm::ParameterSetDescription desc;
  //it does not matter for makePSetDescription whether tag or probe types are used
  desc.addVPSet("tagRangeCuts", VarRangeCut<TagType>::makePSetDescription(), std::vector<edm::ParameterSet>());
  desc.addVPSet("probeRangeCuts", VarRangeCut<TagType>::makePSetDescription(), std::vector<edm::ParameterSet>());
  desc.add<edm::InputTag>("trigEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<edm::InputTag>("tagColl", edm::InputTag());
  desc.add<edm::InputTag>("probeColl", edm::InputTag());
  desc.add<edm::InputTag>("tagVIDCuts", edm::InputTag());
  desc.add<edm::InputTag>("probeVIDCuts", edm::InputTag());
  desc.add<std::vector<std::string> >("tagFilters", std::vector<std::string>());
  desc.add<std::vector<std::string> >("probeFilters", std::vector<std::string>());
  desc.add<bool>(
      "tagFiltersORed",
      true);  //default to OR probe filters (use case is multiple tag triggers, eg Ele27, Ele32, Ele35 tight etc)
  desc.add<bool>(
      "probeFiltersORed",
      false);  //default to AND probe filters (cant think why you would want to OR them but made if configurable just in case)
  desc.add<double>("minTagProbeDR", 0);
  desc.add<double>("minMass");
  desc.add<double>("maxMass");
  desc.add<bool>("requireOpSign");
  desc.addVPSet("histConfigs",
                HLTDQMFilterTnPEffHists<TagType, ProbeType>::makePSetDescriptionHistConfigs(),
                std::vector<edm::ParameterSet>());
  desc.addVPSet("filterConfigs",
                HLTDQMFilterTnPEffHists<TagType, ProbeType>::makePSetDescription(),
                std::vector<edm::ParameterSet>());
  desc.add<std::string>("baseHistName");

  edm::ParameterSetDescription trigEvtFlagDesc;
  trigEvtFlagDesc.add<bool>("andOr", false);
  trigEvtFlagDesc.add<unsigned int>("verbosityLevel", 1);
  trigEvtFlagDesc.add<bool>("andOrDcs", false);
  trigEvtFlagDesc.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi"));
  trigEvtFlagDesc.add<edm::InputTag>("dcsRecordInputTag", edm::InputTag("onlineMetaDataDigis"));
  trigEvtFlagDesc.add<std::vector<int> >("dcsPartitions", {24, 25, 26, 27, 28, 29});
  trigEvtFlagDesc.add<bool>("errorReplyDcs", true);
  trigEvtFlagDesc.add<std::string>("dbLabel", "");
  trigEvtFlagDesc.add<bool>("andOrHlt", true);  //true = OR, false = and
  trigEvtFlagDesc.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT"));
  trigEvtFlagDesc.add<std::vector<std::string> >("hltPaths", {});
  trigEvtFlagDesc.add<std::string>("hltDBKey", "");
  trigEvtFlagDesc.add<bool>("errorReplyHlt", false);
  desc.add<edm::ParameterSetDescription>("sampleTrigRequirements", trigEvtFlagDesc);

  return desc;
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
void HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::bookHists(DQMStore::IBooker& iBooker) {
  for (auto& filterHist : filterHists_)
    filterHist.bookHists(iBooker, histConfigs_);
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
void HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::beginRun(const edm::Run& run,
                                                                                    const edm::EventSetup& setup) {
  if (sampleTrigRequirements_.on())
    sampleTrigRequirements_.initRun(run, setup);
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
void HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::fill(const edm::Event& event,
                                                                                const edm::EventSetup& setup) {
  auto tagCollHandle = getHandle(event, tagToken_);
  auto probeCollHandle = getHandle(event, probeToken_);
  auto trigEvtHandle = getHandle(event, trigEvtToken_);
  auto tagVIDHandle = getHandle(event, tagVIDToken_);
  auto probeVIDHandle = getHandle(event, probeVIDToken_);

  //we need the object collection and trigger info at the minimum
  if (!tagCollHandle.isValid() || !probeCollHandle.isValid() || !trigEvtHandle.isValid())
    return;

  //if GenericTriggerEventFlag is "off", it'll return true regardless
  //if so if its off, we auto pass which is the behaviour we wish to have
  //if its null, we auto fail (because that shouldnt happen)
  if (sampleTrigRequirements_.accept(event, setup) == false)
    return;

  std::vector<edm::Ref<TagCollType> > tagRefs =
      getPassingRefs(tagCollHandle, *trigEvtHandle, tagFilters_, tagFiltersORed_, tagVIDHandle, tagRangeCuts_);

  std::vector<edm::Ref<ProbeCollType> > probeRefs = getPassingRefs(
      probeCollHandle, *trigEvtHandle, probeFilters_, probeFiltersORed_, probeVIDHandle, probeRangeCuts_);

  for (auto& tagRef : tagRefs) {
    float tagEta = tagRef->eta();
    float tagPhi = tagRef->phi();
    for (auto& probeRef : probeRefs) {
      //first check if its the same object via its memory localation
      //note for different collections another method is needed to determine
      //if the probe and tag are the same object just recoed differently
      //suggest dR cut (although mass cut should also help here)
      if (static_cast<const void*>(&*tagRef) == static_cast<const void*>(&*probeRef))
        continue;
      float dR2 = reco::deltaR2(tagEta, tagPhi, probeRef->eta(), probeRef->phi());
      float mass = (tagRef->p4() + probeRef->p4()).mag();
      if ((mass >= minMass_ || minMass_ < 0) && (mass < maxMass_ || maxMass_ < 0) && (dR2 >= minTagProbeDR2_) &&
          (!requireOpSign_ || tagRef->charge() != probeRef->charge())) {
        for (auto& filterHist : filterHists_) {
          filterHist.fillHists(*tagRef, *probeRef, event, setup, *trigEvtHandle);
        }
      }  //end of t&p pair cuts
    }    //end of probe loop
  }      //end of tag loop
}

//yo dawg, I heard you like templates...
//okay this might be a little confusing to the student expected to maintain this
//here we have a templated function inside a templated class as it needs to be able to take probe or tag types
//so this is function of a class HLTDQMTagAndProbeEff<TagType,TagCollType,ProbeType,ProbeCollType>
//hence why it needs to specify that those types even if it doesnt use them
//However also templated to take a type of ObjCollType
//(which we know will either be ProbeCollType or TagCollType but c++ doesnt and therefore it can anything)
//this is why there are two seperate template declarations
template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
template <typename ObjType, typename ObjCollType>
std::vector<edm::Ref<ObjCollType> > HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::getPassingRefs(
    const edm::Handle<ObjCollType>& objCollHandle,
    const trigger::TriggerEvent& trigEvt,
    const std::vector<std::string>& filterNames,
    const bool orFilters,
    const edm::Handle<edm::ValueMap<bool> >& vidHandle,
    const VarRangeCutColl<ObjType>& rangeCuts) {
  std::vector<edm::Ref<ObjCollType> > passingRefs;
  for (size_t objNr = 0; objNr < objCollHandle->size(); objNr++) {
    edm::Ref<ObjCollType> ref(objCollHandle, objNr);
    if (rangeCuts(*ref) && hltdqm::passTrig(ref->eta(), ref->phi(), trigEvt, filterNames, orFilters, hltProcess_) &&
        (vidHandle.isValid() == false || (*vidHandle)[ref] == true)) {
      passingRefs.push_back(ref);
    }
  }
  return passingRefs;
}
#endif
