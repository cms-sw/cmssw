#ifndef __newpf_PFClusterProducer_H__
#define __newpf_PFClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/TopoClusterBuilderFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorFactory.h"

#include <memory>

namespace newpf {
  class PFClusterProducer : public edm::EDProducer {
    typedef TopoClusterBuilderBase TCBB;
    typedef PFClusterBuilderBase PFCBB;
    typedef PFCPositionCalculatorBase PosCalc;
  public:    
    PFClusterProducer(const edm::ParameterSet&);
    ~PFClusterProducer() { }

    virtual void beginLuminosityBlock(const edm::LuminosityBlock&, 
				      const edm::EventSetup&);
    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:
    // inputs
    edm::EDGetTokenT<reco::PFRecHit> _rechitsLabel;
    // options
    const bool _prodTopoClusters;
    // the actual algorithm
    std::unique_ptr<TopoClusterBuilderBase> _topoBuilder;
    std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
    std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  };
}

typedef newpf::PFClusterProducer PFClusterProducerNew;
DEFINE_FWK_MODULE(PFClusterProducerNew);

#endif
