#ifndef EgammaHLTProducers_EgammaHLTHcalPFClusterIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTHcalPFClusterIsolationProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"


namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTHcalPFClusterIsolationProducer : public edm::EDProducer {
 public:
  explicit EgammaHLTHcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~EgammaHLTHcalPFClusterIsolationProducer();    
      
  virtual void produce(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
 private:

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
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
};

#endif
