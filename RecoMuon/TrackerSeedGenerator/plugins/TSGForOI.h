/**
  \class    TSGForOI
  \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside in manner
  \author   Benjamin Radburn-Smith
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"


class TSGForOI : public edm::stream::EDProducer<> {
public:
	explicit TSGForOI(const edm::ParameterSet & iConfig);
	virtual ~TSGForOI();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
	virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;
private:
	/// Labels for input collections
	edm::EDGetTokenT<reco::TrackCollection> src_;

	/// Maximum number of seeds for each L2
	unsigned int numOfMaxSeeds_;

	/// How many layers to try
	unsigned int numOfLayersToTry_;

	/// How many hits to try on same layer
	unsigned int numOfHitsToTry_;

	/// How much to rescale errors from STA for fixed option
	double fixedErrorRescalingForHits_;
	double fixedErrorRescalingForHitless_;

	/// Whether or not to use an automatically calculated SF value
	bool adjustErrorsDyanmicallyForHits_;
	bool adjustErrorsDyanmicallyForHitless_;

	///Estimator used to find dets and TrajectoryMeasurements
	std::string estimatorName_;

	std::string trackerPropagatorName_;
	std::string muonPropagatorName_;
	edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerTag_;
	std::string measurementTrackerName_;

	double minEtaForTEC_, maxEtaForTOB_;

	/// Switch to use hitless seeds or not
	bool useHitlessSeeds_;

	/// Surface used to make a TSOS at the PCA to the beamline
	Plane::PlanePointer dummyPlane_;

	/// KFUpdator defined in constructor
	const TrajectoryStateUpdator* updator_;

	bool foundCompatibleDet_;
	bool analysedL2_;

	unsigned int numSeedsMade_;
	unsigned int layerCount_;

	std::string theCategory;

	edm::ESHandle<MagneticField>          magfield_;
	edm::ESHandle<Propagator>             propagatorAlong_;
	edm::ESHandle<Propagator>             propagatorOpposite_;
	edm::ESHandle<GlobalTrackingGeometry> geometry_;
	edm::Handle<MeasurementTrackerEvent> measurementTracker_;

	/// Function to find seeds on a given layer
	void findSeedsOnLayer(const GeometricSearchDet &layer,
			const TrajectoryStateOnSurface &tsosAtIP,
			const TrajectoryStateOnSurface &tsosAtMuonSystem,
			const Propagator& propagatorAlong,
			const Propagator& propagatorOpposite,
			const reco::TrackRef l2,
			std::auto_ptr<std::vector<TrajectorySeed> >& seeds);

	/// Function used to calculate the dynamic error SF by analysing the L2
	double calculateSFFromL2(const GeometricSearchDet& layer,
			const TrajectoryStateOnSurface &tsosAtMuonSystem,
			const TrajectoryStateOnSurface &tsosOnLayer,
			const Propagator& propagatorOpposite,
//			const reco::Track& track);
			const reco::TrackRef track);

	/// Function to find hits on layers and create seeds from updated TSOS
	int makeSeedsFromHits(const GeometricSearchDet &layer,
			const TrajectoryStateOnSurface &state,
			std::vector<TrajectorySeed> &out,
			const Propagator& propagatorAlong,
			const MeasurementTrackerEvent &mte,
			double errorSF);

	edm::ESHandle<Chi2MeasurementEstimatorBase>   estimator_;
};
