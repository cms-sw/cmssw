#ifndef __newpf_PFMultiDepthClusterProducer_H__
#define __newpf_PFMultiDepthClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

#include <memory>


class PFMultiDepthClusterProducer : public edm::stream::EDProducer<> {
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;
 public:    
  PFMultiDepthClusterProducer(const edm::ParameterSet&);
  ~PFMultiDepthClusterProducer() { }
  
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, 
				    const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  // inputs
  edm::EDGetTokenT<reco::PFClusterCollection> _clustersLabel;
  // options
  // the actual algorithm
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;
};

DEFINE_FWK_MODULE(PFMultiDepthClusterProducer);

#endif
