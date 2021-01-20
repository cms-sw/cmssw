#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLT2L1TkMuonL1TkMuonMuRefDR.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include <cmath>

//
// constructors and destructor
//
HLT2L1TkMuonL1TkMuonMuRefDR::HLT2L1TkMuonL1TkMuonMuRefDR(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes<trigger::TriggerFilterObjectWithRefs>(inputTag1_)),
      inputToken2_(consumes<trigger::TriggerFilterObjectWithRefs>(inputTag2_)),
      minDR_(iConfig.getParameter<double>("MinDR")),
      min_N_(iConfig.getParameter<int>("MinN")),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

HLT2L1TkMuonL1TkMuonMuRefDR::~HLT2L1TkMuonL1TkMuonMuRefDR() {}

void HLT2L1TkMuonL1TkMuonMuRefDR::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  std::vector<edm::InputTag> originTag1(1, edm::InputTag("hltOriginal1"));
  std::vector<edm::InputTag> originTag2(1, edm::InputTag("hltOriginal2"));
  desc.add<std::vector<edm::InputTag>>("originTag1", originTag1);
  desc.add<std::vector<edm::InputTag>>("originTag2", originTag2);
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<double>("MinDR", -1.0);
  desc.add<int>("MinN", 1);

  descriptions.add("HLT2L1TkMuonL1TkMuonMuRefDR", desc);
}

bool HLT2L1TkMuonL1TkMuonMuRefDR::getCollections(edm::Event& iEvent,
                                                 std::vector<l1t::TkMuonRef>& coll1,
                                                 std::vector<l1t::TkMuonRef>& coll2,
                                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  edm::Handle<trigger::TriggerFilterObjectWithRefs> handle1, handle2;
  if (iEvent.getByToken(inputToken1_, handle1) and iEvent.getByToken(inputToken2_, handle2)) {
    // get hold of pre-filtered object collections
    handle1->getObjects(trigger::TriggerObjectType::TriggerL1TkMu, coll1);
    handle2->getObjects(trigger::TriggerObjectType::TriggerL1TkMu, coll2);
    const trigger::size_type n1(coll1.size());
    const trigger::size_type n2(coll2.size());

    if (saveTags()) {
      edm::InputTag tagOld;
      for (unsigned int i = 0; i < originTag1_.size(); ++i) {
        filterproduct.addCollectionTag(originTag1_[i]);
      }
      tagOld = edm::InputTag();
      for (trigger::size_type i1 = 0; i1 != n1; ++i1) {
        const edm::ProductID pid(coll1[i1].id());
        const std::string& label(iEvent.getProvenance(pid).moduleLabel());
        const std::string& instance(iEvent.getProvenance(pid).productInstanceName());
        const std::string& process(iEvent.getProvenance(pid).processName());
        edm::InputTag tagNew(edm::InputTag(label, instance, process));
        if (tagOld.encode() != tagNew.encode()) {
          filterproduct.addCollectionTag(tagNew);
          tagOld = tagNew;
        }
      }
      for (unsigned int i = 0; i < originTag2_.size(); ++i) {
        filterproduct.addCollectionTag(originTag2_[i]);
      }
      tagOld = edm::InputTag();
      for (trigger::size_type i2 = 0; i2 != n2; ++i2) {
        const edm::ProductID pid(coll2[i2].id());
        const std::string& label(iEvent.getProvenance(pid).moduleLabel());
        const std::string& instance(iEvent.getProvenance(pid).productInstanceName());
        const std::string& process(iEvent.getProvenance(pid).processName());
        edm::InputTag tagNew(edm::InputTag(label, instance, process));
        if (tagOld.encode() != tagNew.encode()) {
          filterproduct.addCollectionTag(tagNew);
          tagOld = tagNew;
        }
      }
    }

    return true;
  } else
    return false;
}

bool HLT2L1TkMuonL1TkMuonMuRefDR::computeDR(edm::Event& iEvent, l1t::TkMuonRef& r1, l1t::TkMuonRef& r2) const {
  float muRef1_eta = 0.;
  float muRef1_phi = 0.;
  if (r1->muonDetector() != 3) {
    if (r1->muRef().isNull())
      return false;

    muRef1_eta = r1->muRef()->hwEta() * 0.010875;
    muRef1_phi = static_cast<float>(l1t::MicroGMTConfiguration::calcGlobalPhi(
        r1->muRef()->hwPhi(), r1->muRef()->trackFinderType(), r1->muRef()->processor()));
    muRef1_phi = muRef1_phi * 2. * M_PI / 576.;
  } else {
    if (r1->emtfTrk().isNull())
      return false;

    muRef1_eta = r1->emtfTrk()->Eta();
    muRef1_phi = angle_units::operators::convertDegToRad(r1->emtfTrk()->Phi_glob());
  }
  muRef1_phi = reco::reduceRange(muRef1_phi);

  float muRef2_eta = 0.;
  float muRef2_phi = 0.;
  if (r2->muonDetector() != 3) {
    if (r2->muRef().isNull())
      return false;

    muRef2_eta = r2->muRef()->hwEta() * 0.010875;
    muRef2_phi = static_cast<float>(l1t::MicroGMTConfiguration::calcGlobalPhi(
        r2->muRef()->hwPhi(), r2->muRef()->trackFinderType(), r2->muRef()->processor()));
    muRef2_phi = muRef2_phi * 2. * M_PI / 576.;
  } else {
    if (r2->emtfTrk().isNull())
      return false;

    muRef2_eta = r2->emtfTrk()->Eta();
    muRef2_phi = angle_units::operators::convertDegToRad(r2->emtfTrk()->Phi_glob());
  }
  muRef2_phi = reco::reduceRange(muRef2_phi);

  if (reco::deltaR(muRef1_eta, muRef1_phi, muRef2_eta, muRef2_phi) > minDR_)
    return true;

  return false;
}

// ------------ method called to produce the data  ------------
bool HLT2L1TkMuonL1TkMuonMuRefDR::hltFilter(edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.
  bool accept(false);

  std::vector<l1t::TkMuonRef> coll1;
  std::vector<l1t::TkMuonRef> coll2;

  if (getCollections(iEvent, coll1, coll2, filterproduct)) {
    int n(0);
    l1t::TkMuonRef r1;
    l1t::TkMuonRef r2;

    for (unsigned int i1 = 0; i1 != coll1.size(); i1++) {
      r1 = coll1[i1];
      unsigned int I(0);
      if (same_) {
        I = i1 + 1;
      }
      for (unsigned int i2 = I; i2 != coll2.size(); i2++) {
        r2 = coll2[i2];

        if (!computeDR(iEvent, r1, r2))
          continue;

        n++;
        filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkMu, r1);
        filterproduct.addObject(trigger::TriggerObjectType::TriggerL1TkMu, r2);
      }
    }

    accept = accept || (n >= min_N_);
  }

  return accept;
}

DEFINE_FWK_MODULE(HLT2L1TkMuonL1TkMuonMuRefDR);
