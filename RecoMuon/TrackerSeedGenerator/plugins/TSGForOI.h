#ifndef RecoMuon_TrackerSeedGenerator_TSGForOI_H
#define RecoMuon_TrackerSeedGenerator_TSGForOI_H

/**
  \class    TSGForOI
  \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside in manner
  \author   Benjamin Radburn-Smith, Santiago Folgueras
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

class TSGForOI : public edm::global::EDProducer<> {
    
  public:
    
    explicit TSGForOI(const edm::ParameterSet & iConfig);
    ~TSGForOI() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  private:
    
    /// Labels for input collections
    const edm::EDGetTokenT<reco::TrackCollection> src_;

    /// Maximum number of seeds for each L2
    const unsigned int numOfMaxSeedsParam_;

    /// How many layers to try
    const unsigned int numOfLayersToTry_;

    /// How many hits to try per layer
    const unsigned int numOfHitsToTry_;

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

    /// Switch ON to use Stereo layers instead of using every layer in TEC.
    const bool useStereoLayersInTEC_;

    /// KFUpdator defined in constructor
    const std::unique_ptr<TrajectoryStateUpdator> updator_;

    const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerTag_;

    /// pT, eta ranges and scale factor values
    const double pT1_,pT2_,pT3_;
    const double eta1_,eta2_,eta3_,eta4_,eta5_,eta6_;
    const double SF1_,SF2_,SF3_,SF4_,SF5_,SF6_;

    /// Distance of L2 TSOSs before and after updated with vertex
    const double tsosDiff1_;
    const double tsosDiff2_;

    /// Counters and flags for the implementation
    const std::string propagatorName_;
    const std::string theCategory;

    /// Find seeds on a given layer
    void findSeedsOnLayer(
        const TrackerTopology* tTopo,
        const GeometricSearchDet& layer,
        const TrajectoryStateOnSurface& tsosAtIP,
        const Propagator& propagatorAlong,
        const Propagator& propagatorOpposite,
        const reco::TrackRef l2,
        edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
        edm::Handle<MeasurementTrackerEvent>& measurementTrackerH,
        unsigned int& numSeedsMade,
        unsigned int& hitlessSeedsMade,
        unsigned int& numOfMaxSeeds,
        unsigned int& layerCount,
        std::unique_ptr<std::vector<TrajectorySeed> >& out) const;

    /// Calculate the dynamic error SF by analysing the L2
    double calculateSFFromL2(const reco::TrackRef track) const;

    /// Find hits on layers and create seeds from updated TSOS
    int makeSeedsFromHits(
        const TrackerTopology* tTopo,
        const GeometricSearchDet& layer,
        const TrajectoryStateOnSurface& tsosAtIP,
        std::vector<TrajectorySeed>& out,
        const Propagator& propagatorAlong,
        const MeasurementTrackerEvent& measurementTracker,
        edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
        unsigned int& numSeedsMade,
        const double errorSF,
        const double l2Eta) const;

    /// Find compatability between two TSOSs
    double match_Chi2(const TrajectoryStateOnSurface& tsos1,
                      const TrajectoryStateOnSurface& tsos2) const;
                                          
};

#endif

