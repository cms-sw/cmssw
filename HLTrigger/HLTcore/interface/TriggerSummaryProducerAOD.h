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
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

#include <map>
#include <set>
#include <string>
#include <vector>

#include <functional>
#include "tbb/concurrent_unordered_set.h"

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
struct GlobalInputTags {
  GlobalInputTags(): filterTagsGlobal_(),collectionTagsGlobal_(){ }
  mutable tbb::concurrent_unordered_set<edm::InputTag,InputTagHash> filterTagsGlobal_;
  mutable tbb::concurrent_unordered_set<edm::InputTag,InputTagHash> collectionTagsGlobal_;
};
 
class TriggerSummaryProducerAOD : public edm::stream::EDProducer<edm::GlobalCache<GlobalInputTags>> {
  
 public:
  explicit TriggerSummaryProducerAOD(const edm::ParameterSet&, const GlobalInputTags *);
  ~TriggerSummaryProducerAOD();
  static  void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;
  static  void globalEndJob(const GlobalInputTags *);

  // additional
  static std::unique_ptr<GlobalInputTags> initializeGlobalCache(edm::ParameterSet const&) {
    return std::unique_ptr<GlobalInputTags> (new GlobalInputTags());
  };

  template <typename C>
  void fillTriggerObjectCollections(const edm::Event&, edm::GetterOfProducts<C>& );

  template <typename T>
  void fillTriggerObject(const T& );
  void fillTriggerObject(const l1extra::L1HFRings& );
  void fillTriggerObject(const l1extra::L1EtMissParticle& );
  void fillTriggerObject(const reco::PFMET& );
  void fillTriggerObject(const reco::CaloMET& );
  void fillTriggerObject(const reco::MET& );

  template <typename C>
    void fillFilterObjectMembers(const edm::Event&, const edm::InputTag& tag, const trigger::Vids &, const std::vector<edm::Ref<C> >&);

  template <typename C>
  void fillFilterObjectMember(const int&, const int&, const edm::Ref<C>&);
  void fillFilterObjectMember(const int&, const int&, const edm::Ref<l1extra::L1HFRingsCollection>&);
  void fillFilterObjectMember(const int&, const int&, const edm::Ref<l1extra::L1EtMissParticleCollection>&);
  void fillFilterObjectMember(const int&, const int&, const edm::Ref<reco::PFMETCollection>&);
  void fillFilterObjectMember(const int&, const int&, const edm::Ref<reco::CaloMETCollection>&);
  void fillFilterObjectMember(const int&, const int&, const edm::Ref<reco::METCollection>&);

 private:
  /// process name
  std::string pn_;

  /// InputTag ordering class
  struct OrderInputTag {
    bool ignoreProcess_;
    OrderInputTag(bool ignoreProcess): ignoreProcess_(ignoreProcess) { };
    inline bool operator()(const edm::InputTag& l, const edm::InputTag& r) const {
      int c = l.label().compare(r.label());
      if(0==c) {
	if(ignoreProcess_) {
	  return l.instance()<r.instance();
	}
	c = l.instance().compare(r.instance());
	if(0==c) {
	  return l.process()<r.process();
	}
      }
      return c < 0;
    };
  };
  typedef std::set<edm::InputTag,OrderInputTag> InputTagSet;

  /// list of L3 filter tags
  InputTagSet filterTagsEvent_;
  InputTagSet filterTagsStream_;

  /// list of L3 collection tags
  InputTagSet collectionTagsEvent_;
  InputTagSet collectionTagsStream_;

  /// trigger object collection
  trigger::TriggerObjectCollection toc_;
  std::vector<std::string> tags_;
  /// global map for indices into toc_: offset per input L3 collection
  std::map<edm::ProductID,unsigned int> offset_;

  /// keys
  trigger::Keys keys_;
  /// ids
  trigger::Vids ids_;

  /// packing decision
  std::vector<bool> maskFilters_;

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
};
#endif
