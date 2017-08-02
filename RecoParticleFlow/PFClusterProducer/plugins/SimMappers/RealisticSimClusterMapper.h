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
 public:
 RealisticSimClusterMapper(const edm::ParameterSet& conf,
			 edm::ConsumesCollector& sumes) :
    InitialClusteringStepBase(conf,sumes),
    invisibleFraction_(conf.getParameter<double>("invisibleFraction")),
    exclusiveFraction_(conf.getParameter<double>("exclusiveFraction")),
    maxDistanceFilter_(conf.getParameter<bool>("maxDistanceFilter")),
    maxDistance_(conf.getParameter<double>("maxDistance")),
    useMCFractionsForExclEnergy_(conf.getParameter<bool>("useMCFractionsForExclEnergy"))
    {
      simClusterToken_ = sumes.consumes<SimClusterCollection>(conf.getParameter<edm::InputTag>("simClusterSrc"));
    }
  virtual ~RealisticSimClusterMapper() {}
  RealisticSimClusterMapper(const RealisticSimClusterMapper&) = delete;
  RealisticSimClusterMapper& operator=(const RealisticSimClusterMapper&) = delete;

  virtual void updateEvent(const edm::Event&) override final;
  virtual void update(const edm::EventSetup&) override final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&) override;
  
 private:  
  hgcal::RecHitTools rhtools_;
  const float invisibleFraction_ = 0.3f;
  const float exclusiveFraction_ = 0.7f;
  const bool maxDistanceFilter_ = false;
  const float maxDistance_ = 10.f;
  const bool useMCFractionsForExclEnergy_ = false;
  edm::EDGetTokenT<SimClusterCollection> simClusterToken_;
  edm::Handle<SimClusterCollection> simClusterH_;
  
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  RealisticSimClusterMapper,
		  "RealisticSimClusterMapper");

#endif
