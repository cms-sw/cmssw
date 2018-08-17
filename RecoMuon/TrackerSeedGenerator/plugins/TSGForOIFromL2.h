#ifndef RecoMuon_TrackerSeedGenerator_TSGForOIFromL2_H
#define RecoMuon_TrackerSeedGenerator_TSGForOIFromL2_H

/**
 \class    TSGForOIFromL2
 \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside-in manner
 \author   Benjamin Radburn-Smith, Santiago Folgueras, Bibhuprasad Mahakud, Jan Frederik Schulte (Purdue University, West Lafayette)
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

class TSGForOIFromL2 : public edm::global::EDProducer<> {
 
  public:
 
    explicit TSGForOIFromL2(const edm::ParameterSet & iConfig);
    ~TSGForOIFromL2() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  private:
 
    /// Labels for input collections
    const edm::EDGetTokenT<reco::TrackCollection> src_;

    /// Maximum number of seeds for each L2
    const unsigned int maxSeeds_;

    /// Maximum number of hitless seeds for each L2
    const unsigned int maxHitlessSeeds_;

    /// Maximum number of hitbased seeds for each L2
    const unsigned int maxHitSeeds_;

    /// How many layers to try
    const unsigned int numOfLayersToTry_;

    /// How many hits to try per layer
    const unsigned int numOfHitsToTry_;

    ///L2 valid hit cuts to decide seed creation by both states
    const unsigned int numL2ValidHitsCutAllEta_;
    const unsigned int numL2ValidHitsCutAllEndcap_;

    /// Rescale L2 parameter uncertainties (fixed error vs pT, eta)
    const double fixedErrorRescalingForHits_;
    const double fixedErrorRescalingForHitless_;

    /// Whether or not to use an automatically calculated scale-factor value
    const bool adjustErrorsDynamicallyForHits_;
    const bool adjustErrorsDynamicallyForHitless_;

    /// Estimator used to find dets and TrajectoryMeasurements
    const std::string estimatorName_;

    /// Minimum eta value to activate searching in the TEC
    const double minEtaForTEC_;

    /// Maximum eta value to activate searching in the TOB
    const double maxEtaForTOB_;

    /// Switch ON  (True) : use additional hits for seeds depending on the L2 properties (ignores numOfMaxSeeds_) 
    /// Switch OFF (False): the numOfMaxSeeds_ defines if we will use hitless (numOfMaxSeeds_==1) or hitless+hits (numOfMaxSeeds_>1) 
    const bool useHitLessSeeds_;

    /// KFUpdator defined in constructor
    const std::unique_ptr<TrajectoryStateUpdator> updator_;

    const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerTag_;

    /// pT, eta ranges and scale factor values
    const double pT1_,pT2_,pT3_;
    const double eta1_,eta2_,eta3_,eta4_,eta5_,eta6_,eta7_;
    const double SF1_,SF2_,SF3_,SF4_,SF5_,SF6_;

    /// Distance of L2 TSOSs before and after updated with vertex
    const double tsosDiff1_;
    const double tsosDiff2_;

    /// Counters and flags for the implementation
    const std::string propagatorName_;
    const std::string theCategory_;

    /// Create seeds without hits on a given layer (TOB or TEC)
    void makeSeedsWithoutHits(
        const GeometricSearchDet& layer,
        const TrajectoryStateOnSurface& tsos,
        const Propagator& propagatorAlong,
        edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
        double errorSF,
        unsigned int& hitlessSeedsMade,
        unsigned int& numSeedsMade,
        std::vector<TrajectorySeed>& out) const;

    /// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
    void makeSeedsFromHits(
        const GeometricSearchDet& layer,
        const TrajectoryStateOnSurface& tsos,
        const Propagator& propagatorAlong,
        edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
        edm::Handle<MeasurementTrackerEvent>& measurementTracker,
        double errorSF,
        unsigned int& hitSeedsMade,
        unsigned int& numSeedsMade,
        unsigned int& layerCount,
        std::vector<TrajectorySeed>& out) const;
    
    /// Calculate the dynamic error SF by analysing the L2
    double calculateSFFromL2(const reco::TrackRef track) const;
 
    /// Find compatability between two TSOSs
    double match_Chi2(const TrajectoryStateOnSurface& tsos1,
                      const TrajectoryStateOnSurface& tsos2) const;
 
};

#endif
