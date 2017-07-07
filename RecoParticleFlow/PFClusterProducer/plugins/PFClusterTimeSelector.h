#ifndef RecoParticleFlow_PFClusterProducer_PFClusterTimeSelector_h_
#define RecoParticleFlow_PFClusterProducer_PFClusterTimeSelector_h_

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

class PFClusterTimeSelector : public edm::stream::EDProducer<> {
 public:
  explicit PFClusterTimeSelector(const edm::ParameterSet&);
  ~PFClusterTimeSelector();

  virtual void beginRun(const edm::Run& run, const edm::EventSetup & es);
  
  void produce(edm::Event& iEvent, 
	       const edm::EventSetup& iSetup);


 protected:

  struct CutInfo {
    double depth;
    double minE;
    double maxE;
    double minTime;
    double maxTime;
    bool endcap;

  };

  // ----------access to event data
  edm::EDGetTokenT<reco::PFClusterCollection> clusters_;
  std::vector<CutInfo> cutInfo_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterTimeSelector);

#endif
