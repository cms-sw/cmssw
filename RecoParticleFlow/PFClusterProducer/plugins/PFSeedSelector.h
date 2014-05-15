#ifndef RecoParticleFlow_PFClusterProducer_PFCTRecHitProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFCTRecHitProducer_h_

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

class PFSeedSelector : public edm::EDProducer {
 public:
  explicit PFSeedSelector(const edm::ParameterSet&);
  ~PFSeedSelector();

  virtual void beginRun(const edm::Run& run, const edm::EventSetup & es);
  
  void produce(edm::Event& iEvent, 
	       const edm::EventSetup& iSetup);



 protected:
  // ----------access to event data
  edm::EDGetTokenT<reco::PFRecHitCollection> hits_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFSeedSelector);

#endif
