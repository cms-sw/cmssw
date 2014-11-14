#ifndef EgammaHLTProducers_HLTHcalPFClusterIsolationProducer_h
#define EgammaHLTProducers_HLTHcalPFClusterIsolationProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"


namespace edm {
  class ConfigurationDescriptions;
}

template<typename T1>
class HLTHcalPFClusterIsolationProducer : public edm::EDProducer {

  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef edm::AssociationMap<edm::OneToValue<std::vector<T1>, float > > T1IsolationMap;

 public:
  explicit HLTHcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~HLTHcalPFClusterIsolationProducer();    
      
  virtual void produce(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
 private:

  edm::EDGetTokenT<T1Collection> recoCandidateProducer_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHCAL_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHFEM_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHFHAD_;
  edm::EDGetTokenT<double> rhoProducer_;

  double drMax_;
  double drVetoBarrel_;
  double drVetoEndcap_;
  double etaStripBarrel_;
  double etaStripEndcap_;
  double energyBarrel_;
  double energyEndcap_;
  
  float effectiveAreaBarrel_;
  float effectiveAreaEndcap_;
  bool doRhoCorrection_;
  float rhoScale_;
  float rhoMax_;
  
  bool useHF_;
};

#endif
