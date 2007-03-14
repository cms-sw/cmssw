// ---------------------------------------------------------------------------
// This class performs a simple disentaglment of the matched rechits
// providing a vector of rechits and angles.
//
// Class objects need to be initialized with the event setup. It needs
// to access the tracker geometry to convert the local directions.
// The SeparateHits method requires a reco::TrackInfoRef. It takes it
// by reference. It returns a vector<pair<const TrackingRecHit*,float> >,
// which contains all the hits (matched hits are divided in mono and stereo)
// and the corresponding track angles.
// It also stores the phi and theta angles and the local and global direction
// vectors which can be taken by the corresponding get methods.
//
// M.De Mattia 13/2/2007
// ---------------------------------------------------------------------------

#ifndef AnalysisExamples_SiStripDetectorPerformance_TrackLocalAngleTIF_h
#define AnalysisExamples_SiStripDetectorPerformance_TrackLocalAngleTIF_h

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
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"

class TrackLocalAngleTIF 
{
 public:
  typedef std::vector<std::pair<const TrackingRecHit *, float> > HitAngleAssociation;
  typedef std::vector<std::pair<const TrackingRecHit *, LocalVector > > HitLclDirAssociation;
  typedef std::vector<std::pair<const TrackingRecHit *, GlobalVector> > HitGlbDirAssociation;
  
  TrackLocalAngleTIF();
  
  virtual ~TrackLocalAngleTIF();

  void init ( const edm::EventSetup& es  );

  std::vector<std::pair<const TrackingRecHit*,float> > SeparateHits(reco::TrackInfoRef & trackinforef);

  inline HitAngleAssociation getXZHitAngle() const throw() { 
    return oXZHitAngle; }
  inline HitAngleAssociation getYZHitAngle() const throw() { 
    return oYZHitAngle; }

  inline HitLclDirAssociation getLocalDir() const throw() {
    return oLocalDir; }
  inline HitGlbDirAssociation getGlobalDir() const throw() {
    return oGlobalDir; }

 private:

  HitAngleAssociation oXZHitAngle;
  HitAngleAssociation oYZHitAngle;

  HitLclDirAssociation oLocalDir;
  HitGlbDirAssociation oGlobalDir;

  const TrackerGeometry * _tracker;
  reco::TrackInfo::TrajectoryInfo::const_iterator _tkinfoiter;
};


#endif // AnalysisExamples_SiStripDetectorPerformance_TrackLocalAngleTIF_h
