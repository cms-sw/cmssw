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
    maxDforTimingSquared_(conf.getParameter<double>("maxDforTimingSquared")),
    timeOffset_(conf.getParameter<double>("timeOffset")),
    minNHitsforTiming_(conf.getParameter<unsigned int>("minNHitsforTiming")),
    useMCFractionsForExclEnergy_(conf.getParameter<bool>("useMCFractionsForExclEnergy")),
    calibMinEta_(conf.getParameter<double>("calibMinEta")),
    calibMaxEta_(conf.getParameter<double>("calibMaxEta"))
    {
      simClusterToken_ = sumes.consumes<SimClusterCollection>(conf.getParameter<edm::InputTag>("simClusterSrc"));
      hadronCalib_ = conf.getParameter < std::vector<double> > ("hadronCalib");
      egammaCalib_ = conf.getParameter < std::vector<double> > ("egammaCalib");
    }

  ~RealisticSimClusterMapper() override {}
  RealisticSimClusterMapper(const RealisticSimClusterMapper&) = delete;
  RealisticSimClusterMapper& operator=(const RealisticSimClusterMapper&) = delete;

  void updateEvent(const edm::Event&) final;
  void update(const edm::EventSetup&) final;

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
  const float maxDforTimingSquared_ = 4.0f;
  const float timeOffset_;
  const unsigned int minNHitsforTiming_ = 3;
  const bool useMCFractionsForExclEnergy_ = false;
  const float calibMinEta_ = 1.4;
  const float calibMaxEta_ = 3.0;
  std::vector<double> hadronCalib_;
  std::vector<double> egammaCalib_;

  edm::EDGetTokenT<SimClusterCollection> simClusterToken_;
  edm::Handle<SimClusterCollection> simClusterH_;
  
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  RealisticSimClusterMapper,
		  "RealisticSimClusterMapper");

#endif
