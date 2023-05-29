#include "HLTDoubletDZ.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <cmath>

template <typename T1, typename T2>
HLTDoubletDZ<T1, T2>::HLTDoubletDZ(edm::ParameterSet const& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(edm::EDGetTokenT<reco::ElectronCollection>()),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(iConfig.getParameter<int>("MinPixHitsForDZ")),
      checkSC_(iConfig.getParameter<bool>("checkSC")),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <>
HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoEcalCandidate>::HLTDoubletDZ(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(consumes(iConfig.getParameter<edm::InputTag>("electronTag"))),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(iConfig.getParameter<int>("MinPixHitsForDZ")),
      checkSC_(iConfig.getParameter<bool>("checkSC")),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <>
HLTDoubletDZ<reco::RecoChargedCandidate, reco::RecoEcalCandidate>::HLTDoubletDZ(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(consumes(iConfig.getParameter<edm::InputTag>("electronTag"))),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(iConfig.getParameter<int>("MinPixHitsForDZ")),
      checkSC_(iConfig.getParameter<bool>("checkSC")),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <>
HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoChargedCandidate>::HLTDoubletDZ(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(consumes(iConfig.getParameter<edm::InputTag>("electronTag"))),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(iConfig.getParameter<int>("MinPixHitsForDZ")),
      checkSC_(iConfig.getParameter<bool>("checkSC")),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <>
HLTDoubletDZ<l1t::TrackerMuon, l1t::TrackerMuon>::HLTDoubletDZ(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(edm::EDGetTokenT<reco::ElectronCollection>()),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(0),
      checkSC_(false),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <>
HLTDoubletDZ<l1t::PFTau, l1t::PFTau>::HLTDoubletDZ(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(edm::EDGetTokenT<reco::ElectronCollection>()),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(0),
      checkSC_(false),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <>
HLTDoubletDZ<l1t::HPSPFTau, l1t::HPSPFTau>::HLTDoubletDZ(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      electronToken_(edm::EDGetTokenT<reco::ElectronCollection>()),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      minDR_(iConfig.getParameter<double>("MinDR")),
      minDR2_(minDR_ * minDR_),
      maxDZ_(iConfig.getParameter<double>("MaxDZ")),
      min_N_(iConfig.getParameter<int>("MinN")),
      minPixHitsForDZ_(0),
      checkSC_(false),
      same_(inputTag1_.encode() == inputTag2_.encode())  // same collections to be compared?
{}

template <typename T1, typename T2>
void HLTDoubletDZ<T1, T2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  desc.add<int>("MinPixHitsForDZ", 0);
  desc.add<bool>("checkSC", false);
  descriptions.addWithDefaultLabel(desc);
}

template <>
void HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoEcalCandidate>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<edm::InputTag>("electronTag", edm::InputTag("electronTag"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  desc.add<int>("MinPixHitsForDZ", 0);
  desc.add<bool>("checkSC", false);
  descriptions.addWithDefaultLabel(desc);
}

template <>
void HLTDoubletDZ<reco::RecoChargedCandidate, reco::RecoEcalCandidate>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<edm::InputTag>("electronTag", edm::InputTag("electronTag"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  desc.add<int>("MinPixHitsForDZ", 0);
  desc.add<bool>("checkSC", false);
  descriptions.addWithDefaultLabel(desc);
}

template <>
void HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoChargedCandidate>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<edm::InputTag>("electronTag", edm::InputTag("electronTag"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  desc.add<int>("MinPixHitsForDZ", 0);
  desc.add<bool>("checkSC", false);
  descriptions.addWithDefaultLabel(desc);
}

template <>
void HLTDoubletDZ<l1t::TrackerMuon, l1t::TrackerMuon>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  descriptions.addWithDefaultLabel(desc);
}

template <>
void HLTDoubletDZ<l1t::PFTau, l1t::PFTau>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  descriptions.addWithDefaultLabel(desc);
}

template <>
void HLTDoubletDZ<l1t::HPSPFTau, l1t::HPSPFTau>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<double>("MinDR", -1.0);
  desc.add<double>("MaxDZ", 0.2);
  desc.add<int>("MinN", 1);
  descriptions.addWithDefaultLabel(desc);
}

template <typename T1, typename T2>
bool HLTDoubletDZ<T1, T2>::getCollections(edm::Event const& iEvent,
                                          std::vector<T1Ref>& coll1,
                                          std::vector<T2Ref>& coll2,
                                          trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  edm::Handle<trigger::TriggerFilterObjectWithRefs> handle1, handle2;
  if (iEvent.getByToken(inputToken1_, handle1) and iEvent.getByToken(inputToken2_, handle2)) {
    // get hold of pre-filtered object collections
    handle1->getObjects(triggerType1_, coll1);
    handle2->getObjects(triggerType2_, coll2);
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
        const auto& prov = iEvent.getStableProvenance(pid);
        const std::string& label(prov.moduleLabel());
        const std::string& instance(prov.productInstanceName());
        const std::string& process(prov.processName());
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
        const auto& prov = iEvent.getStableProvenance(pid);
        const std::string& label(prov.moduleLabel());
        const std::string& instance(prov.productInstanceName());
        const std::string& process(prov.processName());
        edm::InputTag tagNew(edm::InputTag(label, instance, process));
        if (tagOld.encode() != tagNew.encode()) {
          filterproduct.addCollectionTag(tagNew);
          tagOld = tagNew;
        }
      }
    }

    return true;
  }

  return false;
}

template <typename T1, typename T2>
bool HLTDoubletDZ<T1, T2>::haveSameSuperCluster(T1 const& c1, T2 const& c2) const {
  return (c1.superCluster().isNonnull() and c2.superCluster().isNonnull() and (c1.superCluster() == c2.superCluster()));
}

template <>
bool HLTDoubletDZ<l1t::TrackerMuon, l1t::TrackerMuon>::haveSameSuperCluster(l1t::TrackerMuon const&,
                                                                            l1t::TrackerMuon const&) const {
  return false;
}

template <>
bool HLTDoubletDZ<l1t::PFTau, l1t::PFTau>::haveSameSuperCluster(l1t::PFTau const&, l1t::PFTau const&) const {
  return false;
}

template <>
bool HLTDoubletDZ<l1t::HPSPFTau, l1t::HPSPFTau>::haveSameSuperCluster(l1t::HPSPFTau const&,
                                                                      l1t::HPSPFTau const&) const {
  return false;
}

template <typename C1, typename C2>
bool HLTDoubletDZ<C1, C2>::passCutMinDeltaR(C1 const& c1, C2 const& c2) const {
  return (minDR_ < 0 or reco::deltaR2(c1, c2) >= minDR2_);
}

template <>
bool HLTDoubletDZ<l1t::TrackerMuon, l1t::TrackerMuon>::passCutMinDeltaR(l1t::TrackerMuon const& m1,
                                                                        l1t::TrackerMuon const& m2) const {
  return (minDR_ < 0 or reco::deltaR2(m1.phEta(), m1.phPhi(), m2.phEta(), m2.phPhi()) >= minDR2_);
}

namespace {

  double getCandidateVZ(reco::RecoChargedCandidate const& cand, bool& isValidVZ, int const minPixHitsForValidVZ) {
    if (minPixHitsForValidVZ > 0) {
      auto const track = cand.track();
      if (not(track.isNonnull() and track.isAvailable() and
              track->hitPattern().numberOfValidPixelHits() >= minPixHitsForValidVZ)) {
        isValidVZ = false;
        return 0;
      }
    }

    isValidVZ = true;
    return cand.vz();
  }

  double getCandidateVZ(reco::RecoEcalCandidate const& cand,
                        bool& isValidVZ,
                        int const minPixHitsForValidVZ,
                        reco::ElectronCollection const& electrons) {
    reco::Electron const* elec_ptr = nullptr;
    for (auto const& elec : electrons) {
      if (elec.superCluster() == cand.superCluster()) {
        elec_ptr = &elec;
      }
    }

    if (elec_ptr == nullptr) {
      // IMPROVE, kept for backward compatibility
      isValidVZ = true;
      return 0;  // equivalent to 'reco::Electron e1; return e1.vz();'
    }

    if (minPixHitsForValidVZ > 0) {
      auto const track = elec_ptr->gsfTrack();
      if (not(track.isNonnull() and track.isAvailable() and
              track->hitPattern().numberOfValidPixelHits() >= minPixHitsForValidVZ)) {
        isValidVZ = false;
        return 0;
      }
    }

    isValidVZ = true;
    return elec_ptr->vz();
  }

  double getCandidateVZ(l1t::HPSPFTau const& cand, bool& isValidVZ) {
    auto const& leadChargedPFCand = cand.leadChargedPFCand();
    if (leadChargedPFCand.isNonnull() and leadChargedPFCand.isAvailable()) {
      auto const& pfTrack = leadChargedPFCand->pfTrack();
      if (pfTrack.isNonnull() and pfTrack.isAvailable()) {
        isValidVZ = true;
        return pfTrack->vertex().z();
      }
    }

    isValidVZ = false;
    return 0;
  }

}  // namespace

template <typename T1, typename T2>
bool HLTDoubletDZ<T1, T2>::computeDZ(edm::Event const&, T1 const& c1, T2 const& c2) const {
  return ((std::abs(c1.vz() - c2.vz()) <= maxDZ_) and passCutMinDeltaR(c1, c2));
}

template <>
bool HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoChargedCandidate>::computeDZ(
    edm::Event const& iEvent, reco::RecoEcalCandidate const& c1, reco::RecoChargedCandidate const& c2) const {
  if (not passCutMinDeltaR(c1, c2))
    return false;

  bool hasValidVZ2 = false;
  auto const vz2 = getCandidateVZ(c2, hasValidVZ2, minPixHitsForDZ_);
  if (not hasValidVZ2)
    return true;

  bool hasValidVZ1 = false;
  auto const& electrons = iEvent.get(electronToken_);
  auto const vz1 = getCandidateVZ(c1, hasValidVZ1, minPixHitsForDZ_, electrons);
  if (not hasValidVZ1)
    return true;

  return (std::abs(vz1 - vz2) <= maxDZ_);
}

template <>
bool HLTDoubletDZ<reco::RecoChargedCandidate, reco::RecoEcalCandidate>::computeDZ(
    edm::Event const& iEvent, reco::RecoChargedCandidate const& c1, reco::RecoEcalCandidate const& c2) const {
  if (not passCutMinDeltaR(c1, c2))
    return false;

  bool hasValidVZ1 = false;
  auto const vz1 = getCandidateVZ(c1, hasValidVZ1, minPixHitsForDZ_);
  if (not hasValidVZ1)
    return true;

  bool hasValidVZ2 = false;
  auto const& electrons = iEvent.get(electronToken_);
  auto const vz2 = getCandidateVZ(c2, hasValidVZ2, minPixHitsForDZ_, electrons);
  if (not hasValidVZ2)
    return true;

  return (std::abs(vz1 - vz2) <= maxDZ_);
}

template <>
bool HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoEcalCandidate>::computeDZ(
    edm::Event const& iEvent, reco::RecoEcalCandidate const& c1, reco::RecoEcalCandidate const& c2) const {
  if (not passCutMinDeltaR(c1, c2))
    return false;

  auto const& electrons = iEvent.get(electronToken_);

  bool hasValidVZ1 = false;
  auto const vz1 = getCandidateVZ(c1, hasValidVZ1, minPixHitsForDZ_, electrons);
  if (not hasValidVZ1)
    return true;

  bool hasValidVZ2 = false;
  auto const vz2 = getCandidateVZ(c2, hasValidVZ2, minPixHitsForDZ_, electrons);
  if (not hasValidVZ2)
    return true;

  return (std::abs(vz1 - vz2) <= maxDZ_);
}

template <>
bool HLTDoubletDZ<reco::RecoChargedCandidate, reco::RecoChargedCandidate>::computeDZ(
    edm::Event const& iEvent, reco::RecoChargedCandidate const& c1, reco::RecoChargedCandidate const& c2) const {
  if (not passCutMinDeltaR(c1, c2))
    return false;

  bool hasValidVZ1 = false;
  auto const vz1 = getCandidateVZ(c1, hasValidVZ1, minPixHitsForDZ_);
  if (not hasValidVZ1)
    return true;

  bool hasValidVZ2 = false;
  auto const vz2 = getCandidateVZ(c2, hasValidVZ2, minPixHitsForDZ_);
  if (not hasValidVZ2)
    return true;

  return (std::abs(vz1 - vz2) <= maxDZ_);
}

template <>
bool HLTDoubletDZ<l1t::TrackerMuon, l1t::TrackerMuon>::computeDZ(edm::Event const& iEvent,
                                                                 l1t::TrackerMuon const& c1,
                                                                 l1t::TrackerMuon const& c2) const {
  return ((std::abs(c1.phZ0() - c2.phZ0()) <= maxDZ_) and passCutMinDeltaR(c1, c2));
}

template <>
bool HLTDoubletDZ<l1t::HPSPFTau, l1t::HPSPFTau>::computeDZ(edm::Event const& iEvent,
                                                           l1t::HPSPFTau const& c1,
                                                           l1t::HPSPFTau const& c2) const {
  if (not passCutMinDeltaR(c1, c2))
    return false;

  bool hasValidVZ1 = false;
  auto const vz1 = getCandidateVZ(c1, hasValidVZ1);
  if (not hasValidVZ1)
    return false;

  bool hasValidVZ2 = false;
  auto const vz2 = getCandidateVZ(c2, hasValidVZ2);
  if (not hasValidVZ2)
    return false;

  return (std::abs(vz1 - vz2) <= maxDZ_);
}

template <typename T1, typename T2>
bool HLTDoubletDZ<T1, T2>::hltFilter(edm::Event& iEvent,
                                     edm::EventSetup const& iSetup,
                                     trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.
  std::vector<T1Ref> coll1;
  std::vector<T2Ref> coll2;

  if (getCollections(iEvent, coll1, coll2, filterproduct)) {
    int n(0);
    T1Ref r1;
    T2Ref r2;

    for (unsigned int i1 = 0; i1 < coll1.size(); ++i1) {
      r1 = coll1[i1];
      unsigned int const I = same_ ? i1 + 1 : 0;
      for (unsigned int i2 = I; i2 < coll2.size(); ++i2) {
        r2 = coll2[i2];

        if (checkSC_ and haveSameSuperCluster(*r1, *r2)) {
          continue;
        }

        if (not computeDZ(iEvent, *r1, *r2)) {
          continue;
        }

        n++;
        filterproduct.addObject(triggerType1_, r1);
        filterproduct.addObject(triggerType2_, r2);
      }
    }

    return (n >= min_N_);
  }

  return false;
}

using HLT2ElectronElectronDZ = HLTDoubletDZ<reco::Electron, reco::Electron>;
using HLT2MuonMuonDZ = HLTDoubletDZ<reco::RecoChargedCandidate, reco::RecoChargedCandidate>;
using HLT2ElectronMuonDZ = HLTDoubletDZ<reco::Electron, reco::RecoChargedCandidate>;
using HLT2PhotonPhotonDZ = HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoEcalCandidate>;
using HLT2MuonPhotonDZ = HLTDoubletDZ<reco::RecoChargedCandidate, reco::RecoEcalCandidate>;
using HLT2PhotonMuonDZ = HLTDoubletDZ<reco::RecoEcalCandidate, reco::RecoChargedCandidate>;
using HLT2L1TkMuonL1TkMuonDZ = HLTDoubletDZ<l1t::TrackerMuon, l1t::TrackerMuon>;
using HLT2L1PFTauL1PFTauDZ = HLTDoubletDZ<l1t::PFTau, l1t::PFTau>;
using HLT2L1HPSPFTauL1HPSPFTauDZ = HLTDoubletDZ<l1t::HPSPFTau, l1t::HPSPFTau>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLT2ElectronElectronDZ);
DEFINE_FWK_MODULE(HLT2MuonMuonDZ);
DEFINE_FWK_MODULE(HLT2ElectronMuonDZ);
DEFINE_FWK_MODULE(HLT2PhotonPhotonDZ);
DEFINE_FWK_MODULE(HLT2PhotonMuonDZ);
DEFINE_FWK_MODULE(HLT2MuonPhotonDZ);
DEFINE_FWK_MODULE(HLT2L1TkMuonL1TkMuonDZ);
DEFINE_FWK_MODULE(HLT2L1PFTauL1PFTauDZ);
DEFINE_FWK_MODULE(HLT2L1HPSPFTauL1HPSPFTauDZ);
