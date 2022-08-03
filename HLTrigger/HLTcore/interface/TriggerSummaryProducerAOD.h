#ifndef HLTcore_TriggerSummaryProducerAOD_h
#define HLTcore_TriggerSummaryProducerAOD_h

/** \class TriggerSummaryProducerAOD
 *
 *
 *  This class is an EDProducer making the HLT summary object for AOD
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#include <functional>
#include "oneapi/tbb/concurrent_unordered_set.h"
#include <regex>

namespace edm {
  class EventSetup;
}

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

/// GlobalCache
struct InputTagHash {
  std::size_t operator()(const edm::InputTag& inputTag) const {
    std::hash<std::string> Hash;
    // bit-wise xor
    return Hash(inputTag.label()) ^ Hash(inputTag.instance()) ^ Hash(inputTag.process());
  }
};
class TriggerSummaryProducerAOD : public edm::global::EDProducer<> {
public:
  explicit TriggerSummaryProducerAOD(const edm::ParameterSet&);
  ~TriggerSummaryProducerAOD() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void endJob() override;

private:
  /// InputTag ordering class
  struct OrderInputTag {
    bool ignoreProcess_;
    OrderInputTag(bool ignoreProcess) : ignoreProcess_(ignoreProcess){};
    inline bool operator()(const edm::InputTag& l, const edm::InputTag& r) const {
      int c = l.label().compare(r.label());
      if (0 == c) {
        if (ignoreProcess_) {
          return l.instance() < r.instance();
        }
        c = l.instance().compare(r.instance());
        if (0 == c) {
          return l.process() < r.process();
        }
      }
      return c < 0;
    };
  };

  using ProductIDtoIndex = std::map<edm::ProductID, unsigned int>;
  using InputTagSet = std::set<edm::InputTag, OrderInputTag>;
  template <typename C>
  void fillTriggerObjectCollections(trigger::TriggerObjectCollection&,
                                    ProductIDtoIndex&,
                                    std::vector<std::string>&,
                                    trigger::Keys&,
                                    const edm::Event&,
                                    const edm::GetterOfProducts<C>&,
                                    const InputTagSet&) const;

  template <typename T>
  void fillTriggerObject(trigger::TriggerObjectCollection&, const T&) const;
  void fillTriggerObject(trigger::TriggerObjectCollection&, const l1extra::L1HFRings&) const;
  void fillTriggerObject(trigger::TriggerObjectCollection&, const l1extra::L1EtMissParticle&) const;
  void fillTriggerObject(trigger::TriggerObjectCollection&, const reco::PFMET&) const;
  void fillTriggerObject(trigger::TriggerObjectCollection&, const reco::CaloMET&) const;
  void fillTriggerObject(trigger::TriggerObjectCollection&, const reco::MET&) const;

  template <typename C>
  void fillFilterObjectMembers(const edm::Event&,
                               const edm::InputTag& tag,
                               const trigger::Vids&,
                               const std::vector<edm::Ref<C>>&,
                               const ProductIDtoIndex&,
                               trigger::Keys& keys,
                               trigger::Vids& oIds) const;

  template <typename C>
  void fillFilterObjectMember(trigger::Keys& keys, trigger::Vids& ids, const int&, const int&, const edm::Ref<C>&) const;
  void fillFilterObjectMember(trigger::Keys& keys,
                              trigger::Vids& ids,
                              const int&,
                              const int&,
                              const edm::Ref<l1extra::L1HFRingsCollection>&) const;
  void fillFilterObjectMember(trigger::Keys& keys,
                              trigger::Vids& ids,
                              const int&,
                              const int&,
                              const edm::Ref<l1extra::L1EtMissParticleCollection>&) const;
  void fillFilterObjectMember(
      trigger::Keys& keys, trigger::Vids& ids, const int&, const int&, const edm::Ref<reco::PFMETCollection>&) const;
  void fillFilterObjectMember(
      trigger::Keys& keys, trigger::Vids& ids, const int&, const int&, const edm::Ref<reco::CaloMETCollection>&) const;
  void fillFilterObjectMember(
      trigger::Keys& keys, trigger::Vids& ids, const int&, const int&, const edm::Ref<reco::METCollection>&) const;

  /// throw on error
  const bool throw_;
  /// process name
  std::string pn_;
  /// module labels which should be avoided
  std::vector<std::regex> moduleLabelPatternsToMatch_;
  std::vector<std::regex> moduleLabelPatternsToSkip_;

  /// list of L3 filter tags
  mutable tbb::concurrent_unordered_set<edm::InputTag, InputTagHash> filterTagsGlobal_;

  /// list of L3 collection tags
  mutable tbb::concurrent_unordered_set<edm::InputTag, InputTagHash> collectionTagsGlobal_;

  /// trigger object collection
  //trigger::TriggerObjectCollection toc_;
  //std::vector<std::string> tags_;
  /// global map for indices into toc_: offset per input L3 collection
  //std::map<edm::ProductID, unsigned int> offset_;

  /// keys
  //trigger::Keys keys_;
  /// ids
  //trigger::Vids ids_;

  /// packing decision
  //std::vector<bool> maskFilters_;

  edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs> getTriggerFilterObjectWithRefs_;
  edm::GetterOfProducts<reco::RecoEcalCandidateCollection> getRecoEcalCandidateCollection_;
  edm::GetterOfProducts<reco::ElectronCollection> getElectronCollection_;
  edm::GetterOfProducts<reco::RecoChargedCandidateCollection> getRecoChargedCandidateCollection_;
  edm::GetterOfProducts<reco::CaloJetCollection> getCaloJetCollection_;
  edm::GetterOfProducts<reco::CompositeCandidateCollection> getCompositeCandidateCollection_;
  edm::GetterOfProducts<reco::METCollection> getMETCollection_;
  edm::GetterOfProducts<reco::CaloMETCollection> getCaloMETCollection_;
  edm::GetterOfProducts<reco::PFMETCollection> getPFMETCollection_;
  edm::GetterOfProducts<reco::IsolatedPixelTrackCandidateCollection> getIsolatedPixelTrackCandidateCollection_;
  edm::GetterOfProducts<l1extra::L1EmParticleCollection> getL1EmParticleCollection_;
  edm::GetterOfProducts<l1extra::L1MuonParticleCollection> getL1MuonParticleCollection_;
  edm::GetterOfProducts<l1extra::L1JetParticleCollection> getL1JetParticleCollection_;
  edm::GetterOfProducts<l1extra::L1EtMissParticleCollection> getL1EtMissParticleCollection_;
  edm::GetterOfProducts<l1extra::L1HFRingsCollection> getL1HFRingsCollection_;
  edm::GetterOfProducts<reco::PFJetCollection> getPFJetCollection_;
  edm::GetterOfProducts<reco::PFTauCollection> getPFTauCollection_;
  edm::GetterOfProducts<l1t::MuonBxCollection> getL1TMuonParticleCollection_;
  edm::GetterOfProducts<l1t::MuonShowerBxCollection> getL1TMuonShowerParticleCollection_;
  edm::GetterOfProducts<l1t::EGammaBxCollection> getL1TEGammaParticleCollection_;
  edm::GetterOfProducts<l1t::JetBxCollection> getL1TJetParticleCollection_;
  edm::GetterOfProducts<l1t::TauBxCollection> getL1TTauParticleCollection_;
  edm::GetterOfProducts<l1t::EtSumBxCollection> getL1TEtSumParticleCollection_;
  edm::GetterOfProducts<l1t::TrackerMuonCollection> getL1TTkMuonCollection_;
  edm::GetterOfProducts<l1t::TkElectronCollection> getL1TTkElectronCollection_;
  edm::GetterOfProducts<l1t::TkEmCollection> getL1TTkEmCollection_;
  edm::GetterOfProducts<l1t::PFJetCollection> getL1TPFJetCollection_;
  edm::GetterOfProducts<l1t::PFTauCollection> getL1TPFTauCollection_;
  edm::GetterOfProducts<l1t::HPSPFTauCollection> getL1THPSPFTauCollection_;
  edm::GetterOfProducts<l1t::PFTrackCollection> getL1TPFTrackCollection_;
};
#endif
