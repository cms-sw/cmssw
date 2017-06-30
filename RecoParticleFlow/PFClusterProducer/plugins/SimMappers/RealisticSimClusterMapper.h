#ifndef __RecoParticleFlow_PFClusterProducer_RealisticSimClusterMapper_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticSimClusterMapper_H__
/////////////////////////
// Author: Felice Pantaleo
// Date:   30/06/2017
// Email: felice@cern.ch
/////////////////////////
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

class RealisticSimClusterMapper : public InitialClusteringStepBase {
  typedef RealisticSimClusterMapper B2DGT;
 public:
 RealisticSimClusterMapper(const edm::ParameterSet& conf,
			 edm::ConsumesCollector& sumes) :
    InitialClusteringStepBase(conf,sumes),
    _invisibleFraction(conf.getParameter<double>("invisibleFraction")),
    _exclusiveFraction(conf.getParameter<double>("exclusiveFraction")),
    _useMCFractionsForExclEnergy(conf.getParameter<bool>("useMCFractionsForExclEnergy"))
    {
      _simClusterToken = sumes.consumes<SimClusterCollection>(conf.getParameter<edm::InputTag>("simClusterSrc"));


    }
  virtual ~RealisticSimClusterMapper() {}
  RealisticSimClusterMapper(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  virtual void updateEvent(const edm::Event&) override final;
  virtual void update(const edm::EventSetup&) override final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&) override;
  
 private:  
  hgcal::RecHitTools _rhtools;
  const float _invisibleFraction = 0.3f;
  const float _exclusiveFraction = 0.7f;
  const bool _useMCFractionsForExclEnergy = false;
  edm::EDGetTokenT<SimClusterCollection> _simClusterToken;
  edm::Handle<SimClusterCollection> _simClusterH;
  
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  RealisticSimClusterMapper,
		  "RealisticSimClusterMapper");

#endif
