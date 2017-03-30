#ifndef RecoMuon_TrackerSeedGenerator_TSGForOI_H
#define RecoMuon_TrackerSeedGenerator_TSGForOI_H

/**
  \class    TSGForOI
  \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside in manner
  \author   Benjamin Radburn-Smith, Santiago Folgueras
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

class TSGForOI : public edm::stream::EDProducer<> {
public:
	explicit TSGForOI(const edm::ParameterSet & iConfig);
	virtual ~TSGForOI();
	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
	/// Labels for input collections
	const edm::EDGetTokenT<reco::TrackCollection> src_;

	/// Maximum number of seeds for each L2
	unsigned int numOfMaxSeeds_;

	/// How many layers to try
	const unsigned int numOfLayersToTry_;

	/// How many hits to try per layer
	const unsigned int numOfHitsToTry_;

	/// How much to rescale errors from the L2 (fixed error vs pT, eta)
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

	/// Surface used to make a TSOS at the PCA to the beamline
	Plane::PlanePointer dummyPlane_;

	/// KFUpdator defined in constructor
	std::unique_ptr<TrajectoryStateUpdator> updator_;

	const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerTag_;

	/// pT, eta ranges and scale factor values
	const double pT1_,pT2_,pT3_;
	const double eta1_,eta2_;
	const double SF1_,SF2_,SF3_,SF4_,SF5_;

	/// Distance of TSOSs to trigger using hits or not
	const double tsosDiff_;

	/// Counters and flags for the implementation
	bool analysedL2_;
	bool foundHitlessSeed_;
	unsigned int numSeedsMade_;
	unsigned int layerCount_;

	std::string theCategory;
	edm::ESHandle<MagneticField>          magfield_;
	edm::ESHandle<Propagator>             propagatorAlong_;
	edm::ESHandle<Propagator>             propagatorOpposite_;
	edm::ESHandle<GlobalTrackingGeometry> geometry_;
	edm::Handle<MeasurementTrackerEvent>  measurementTracker_;
	edm::ESHandle<Chi2MeasurementEstimatorBase>   estimator_;

	/// Function to find seeds on a given layer
	void findSeedsOnLayer(const GeometricSearchDet &layer,
			      const TrajectoryStateOnSurface &tsosAtIP,
			      const Propagator& propagatorAlong,
			      const Propagator& propagatorOpposite,
			      const reco::TrackRef l2,
			      std::unique_ptr<std::vector<TrajectorySeed> >& seeds);

	/// Function used to calculate the dynamic error SF by analysing the L2
	double calculateSFFromL2(const reco::TrackRef track);

	/// Function to find hits on layers and create seeds from updated TSOS
	int makeSeedsFromHits(const GeometricSearchDet &layer,
			const TrajectoryStateOnSurface &state,
			std::vector<TrajectorySeed> &out,
			const Propagator& propagatorAlong,
			const MeasurementTrackerEvent &mte,
			double errorSF);


};

#endif
