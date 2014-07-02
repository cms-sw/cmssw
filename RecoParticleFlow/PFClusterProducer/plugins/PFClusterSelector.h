#ifndef RecoParticleFlow_PFClusterProducer_PFCTRecHitProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFCTRecHitProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

class PFClusterSelector : public edm::stream::EDProducer<> {
 public:
  explicit PFClusterSelector(const edm::ParameterSet&);
  ~PFClusterSelector();

  virtual void beginRun(const edm::Run& run, const edm::EventSetup & es);
  
  void produce(edm::Event& iEvent, 
	       const edm::EventSetup& iSetup);



 protected:
  // ----------access to event data
  edm::EDGetTokenT<reco::PFClusterCollection> clusters_;
  std::vector<double> energyRanges_;
  std::vector<double> timingCutsLowBarrel_;
  std::vector<double> timingCutsHighBarrel_;
  std::vector<double> timingCutsLowEndcap_;
  std::vector<double> timingCutsHighEndcap_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterSelector);

#endif
