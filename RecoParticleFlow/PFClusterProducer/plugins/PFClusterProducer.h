#ifndef __newpf_PFClusterProducer_H__
#define __newpf_PFClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/TopoClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

#include <memory>


class PFClusterProducer : public edm::EDProducer {
  typedef RecHitCleanerBase RHCB;
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
  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;
  // options
  const bool _prodTopoClusters;
  // the actual algorithm
  std::vector<std::unique_ptr<RecHitCleanerBase> > _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<TopoClusterBuilderBase> _topoBuilder;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;
};

DEFINE_FWK_MODULE(PFClusterProducer);

#endif
