#ifndef CalibTracker_SiSitripLorentzAngle_TrackLocalAngle_h
#define CalibTracker_SiSitripLorentzAngle_TrackLocalAngle_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"


/**
*
* Class to refit track to obtain local position and direction on a pixel module
*
*/

class TrackLocalAngle 
{
	public:
		
		struct Trackhit{
			float x;
			float y;
			double alpha;
			double beta;
			double gamma;
		};
	
		explicit TrackLocalAngle(const edm::ParameterSet& conf);
	
		virtual ~TrackLocalAngle();
		
		void init(const edm::Event& e,const edm::EventSetup& c);
	
		/**
		* rebuilds the trajectory 
		*/
		std::vector<Trajectory> buildTrajectory(const reco::Track & theT);
		
		/** 
		* returns a vector of the SiPixelRecHits and the corresponding local track position and direciton
		* !!!pointers to the SiPixelRecHits have to be deleted after calling this method !!!
		*/
		std::vector<std::pair<SiPixelRecHit* , Trackhit> > findPixelParameters(const reco::Track & theT);
	
	private:
		
		edm::ParameterSet conf_;
	
		bool seed_plus;
		const Propagator  *thePropagator;
		const Propagator  *thePropagatorOp;
		KFUpdator *theUpdator;
		Chi2MeasurementEstimator *theEstimator;
		const TransientTrackingRecHitBuilder *RHBuilder;
		const KFTrajectorySmoother * theSmoother;
		const TrajectoryFitter * theFitter;
		const TrackerGeometry * tracker;
		const MagneticField * magfield;
		TrajectoryStateTransform tsTransform;
};


#endif
