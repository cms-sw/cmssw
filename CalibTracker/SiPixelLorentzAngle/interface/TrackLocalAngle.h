#ifndef CalibTracker_SiSitripLorentzAngle_TrackLocalAngle_h
#define CalibTracker_SiSitripLorentzAngle_TrackLocalAngle_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
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
  
  //  typedef TransientTrackingRecHit::ConstRecHitPointer    ConstRecHitPointer;
  //typedef TransientTrackingRecHit::RecHitPointer         RecHitPointer;
  //typedef ConstReferenceCountingPointer<TransientTrackingRecHit> ConstRecHitPointer;
  
  explicit TrackLocalAngle(const edm::ParameterSet& conf);
  
  virtual ~TrackLocalAngle();
  void init(const edm::Event& e,const edm::EventSetup& c);

  std::vector<std::pair<const TrackingRecHit*,float> > findtrackangle(const TrajectorySeed& seed,
										       const reco::Track & theT);

  std::vector<std::pair<const TrackingRecHit *,float> > findtrackangle(const reco::Track & theT);
	std::vector<Trajectory> buildTrajectory(const reco::Track & theT);
	std::vector<std::pair<const TrackingRecHit *, Trackhit> > findPixelParameters(const reco::Track & theT);
	
  TrajectoryStateOnSurface  startingTSOS(const TrajectorySeed& seed)const;

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
