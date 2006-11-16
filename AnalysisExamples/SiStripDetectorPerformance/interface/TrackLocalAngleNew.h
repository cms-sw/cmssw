#ifndef AnalysisExamples_SiStripDetectorPerformance_TrackLocalAngleNew_h
#define AnalysisExamples_SiStripDetectorPerformance_TrackLocalAngleNew_h

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

class TrackLocalAngleNew 
{
 public:
  typedef std::vector<std::pair<const TrackingRecHit *, float> > HitAngleAssociation;
  typedef std::vector<std::pair<const TrackingRecHit *, LocalVector > > HitLclDirAssociation;
  typedef std::vector<std::pair<const TrackingRecHit *, GlobalVector> > HitGlbDirAssociation;
  
  //  typedef TransientTrackingRecHit::ConstRecHitPointer    ConstRecHitPointer;
  //typedef TransientTrackingRecHit::RecHitPointer         RecHitPointer;
  //typedef ConstReferenceCountingPointer<TransientTrackingRecHit> ConstRecHitPointer;
  
  explicit TrackLocalAngleNew(const edm::ParameterSet& conf);
  
  virtual ~TrackLocalAngleNew();
  void init(const edm::Event& e,const edm::EventSetup& c);

  std::vector<std::pair<const TrackingRecHit*,float> > findtrackangle(const TrajectorySeed& seed,
										       const reco::Track & theT);

  std::vector<std::pair<const TrackingRecHit *,float> > findtrackangle(const reco::Track & theT);
  std::vector<std::pair<const TrackingRecHit *,float> > buildTrack(TransientTrackingRecHit::RecHitContainer& hits,
								   const TrajectoryStateOnSurface& theTSOS,
								   const TrajectorySeed& seed);
  TrajectoryStateOnSurface  startingTSOS(const TrajectorySeed& seed)const;

  inline HitAngleAssociation getXZHitAngle() const throw() { 
    return oXZHitAngle; }
  inline HitAngleAssociation getYZHitAngle() const throw() { 
    return oYZHitAngle; }

  inline HitLclDirAssociation getLocalDir() const throw() {
    return oLocalDir; }
  inline HitGlbDirAssociation getGlobalDir() const throw() {
    return oGlobalDir; }

 private:
  edm::ParameterSet conf_;
  
  bool seed_plus;
  const Propagator  *thePropagator;
  const Propagator  *thePropagatorOp;
  KFUpdator *theUpdator;
  Chi2MeasurementEstimator *theEstimator;
  const TransientTrackingRecHitBuilder *RHBuilder;
  //  const KFTrajectorySmoother * theSmoother;
  const TrajectoryFitter * theFitter;
  const TrackerGeometry * tracker;
  const MagneticField * magfield;
  TrajectoryStateTransform tsTransform;
  //  TransientTrackingRecHit::RecHitContainer hits;

  HitAngleAssociation oXZHitAngle;
  HitAngleAssociation oYZHitAngle;

  HitLclDirAssociation oLocalDir;
  HitGlbDirAssociation oGlobalDir;
};


#endif // AnalysisExamples_SiStripDetectorPerformance_TrackLocalAngleNew_h
