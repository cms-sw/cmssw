#ifndef RecoParticleFlow_PFClusterProducer_PFArborLinker_h_
#define RecoParticleFlow_PFClusterProducer_PFArborLinker_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "RecoParticleFlow/PFClusterProducer/interface/Arbor.hh"

class PFArborLinker : public edm::EDProducer {
 public:
  explicit PFArborLinker(const edm::ParameterSet&);
  ~PFArborLinker();

  virtual void beginRun(const edm::Run& run, const edm::EventSetup & es);
  
  void produce(edm::Event& iEvent, 
	       const edm::EventSetup& iSetup);



 protected:
  // ----------access to event data
  edm::EDGetTokenT<reco::PFRecHitCollection> hits_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFArborLinker);

#endif
