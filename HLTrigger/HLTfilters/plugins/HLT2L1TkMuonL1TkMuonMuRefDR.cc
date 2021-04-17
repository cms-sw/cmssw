#include <cmath>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

#include "HLT2L1TkMuonL1TkMuonMuRefDR.h"

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
      same_(inputTag1_.encode() == inputTag2_.encode()) {}  // same collections to be compared?

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

  descriptions.add("hlt2L1TkMuonL1TkMuonMuRefDR", desc);
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

std::pair<float, float> HLT2L1TkMuonL1TkMuonMuRefDR::convertEtaPhi(l1t::TkMuonRef& tkmu) const {
  float muRefEta = 0.;
  float muRefPhi = 0.;

  if (tkmu->muonDetector() != emtfRegion_) {
    if (tkmu->muRef().isNull())
      return std::make_pair(muRefEta, muRefPhi);

    muRefEta = tkmu->muRef()->hwEta() * etaScale_;
    muRefPhi = static_cast<float>(l1t::MicroGMTConfiguration::calcGlobalPhi(
        tkmu->muRef()->hwPhi(), tkmu->muRef()->trackFinderType(), tkmu->muRef()->processor()));
    muRefPhi = muRefPhi * phiScale_;
  } else {
    if (tkmu->emtfTrk().isNull())
      return std::make_pair(muRefEta, muRefPhi);

    muRefEta = tkmu->emtfTrk()->Eta();
    muRefPhi = angle_units::operators::convertDegToRad(tkmu->emtfTrk()->Phi_glob());
  }
  muRefPhi = reco::reduceRange(muRefPhi);

  return std::make_pair(muRefEta, muRefPhi);
}

bool HLT2L1TkMuonL1TkMuonMuRefDR::computeDR(edm::Event& iEvent, l1t::TkMuonRef& r1, l1t::TkMuonRef& r2) const {
  if (minDR_ < 0.)
    return true;

  auto [eta1, phi1] = convertEtaPhi(r1);
  auto [eta2, phi2] = convertEtaPhi(r2);
  return (reco::deltaR2(eta1, phi1, eta2, phi2) > minDR_ * minDR_);
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
