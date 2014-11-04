#ifndef L3MuonIsolationProducer_MuonHLTHcalPFClusterIsolationProducer_h
#define L3MuonIsolationProducer_MuonHLTHcalPFClusterIsolationProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"


namespace edm {
  class ConfigurationDescriptions;
}

class MuonHLTHcalPFClusterIsolationProducer : public edm::EDProducer {
 public:
  explicit MuonHLTHcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~MuonHLTHcalPFClusterIsolationProducer();    
      
  virtual void produce(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
 private:

  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> recoChargedCandidateProducer_;
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
