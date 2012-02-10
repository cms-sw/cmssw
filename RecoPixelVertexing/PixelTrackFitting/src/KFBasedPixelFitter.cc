#include "RecoPixelVertexing/PixelTrackFitting/interface/KFBasedPixelFitter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "CircleFromThreePoints.h"

#include <sstream>
template <class T> inline T sqr( T t) {return t*t;}

KFBasedPixelFitter::KFBasedPixelFitter( const edm::ParameterSet& cfg)
 :  
    thePropagatorLabel(cfg.getParameter<std::string>("propagator")),
    thePropagatorOppositeLabel(cfg.getParameter<std::string>("propagatorOpposite")),
    theUseBeamSpot(cfg.getParameter<bool>("useBeamSpotConstraint")),
    theTTRHBuilderName(cfg.getParameter<std::string>("TTRHBuilder")) 
{ 
  if (theUseBeamSpot) theBeamSpot = cfg.getParameter<edm::InputTag>("beamSpotConstraint");
}

reco::Track* KFBasedPixelFitter::run(
    const edm::EventSetup& es,
    const std::vector<const TrackingRecHit * > & hits,
    const TrackingRegion & region) const
{
  int nhits = hits.size();
  if (nhits <2) return 0;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  edm::ESHandle<MagneticField> field;
  es.get<IdealMagneticFieldRecord>().get(field);

  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhb;
  es.get<TransientRecHitRecord>().get( theTTRHBuilderName, ttrhb);

  float ptMin = region.ptMin();

  const GlobalPoint vertexPos = region.origin();
  GlobalError vertexErr( sqr(region.originRBound()), 0, sqr(region.originRBound()), 0, 0, sqr(region.originZBound()));

  std::vector<GlobalPoint> points(nhits);
  points[0] = tracker->idToDet(hits[0]->geographicalId())->toGlobal(hits[0]->localPosition());
  points[1] = tracker->idToDet(hits[1]->geographicalId())->toGlobal(hits[1]->localPosition());
  points[2] = tracker->idToDet(hits[2]->geographicalId())->toGlobal(hits[2]->localPosition());

  //
  //initial Kinematics
  //
  GlobalVector initMom;
  int charge;
  float theta;
  CircleFromThreePoints circle(points[0], points[1], points[2]);
  if (circle.curvature() > 1.e-4) {
    float invPt = PixelRecoUtilities::inversePt( circle.curvature(), es);
    float valPt = 1.f/invPt;
    float chargeTmp =    (points[1].x()-points[0].x())*(points[2].y()-points[1].y())
                       - (points[1].y()-points[0].y())*(points[2].x()-points[1].x()); 
    int charge =  (chargeTmp>0) ? -1 : 1;
    float valPhi = (charge>0) ? std::atan2(circle.center().x(),-circle.center().y()) :  std::atan2(-circle.center().x(),circle.center().y());
    theta = GlobalVector(points[1]-points[0]).theta();
    initMom = GlobalVector(valPt*cos(valPhi), valPt*sin(valPhi), valPt/tan(theta)); 
  } 
  else {
    initMom = GlobalVector(points[1]-points[0]);
    initMom *= 10000./initMom.perp();
    charge = 1;
    theta = initMom.theta();
  }
  GlobalTrajectoryParameters initialKine(vertexPos, initMom, TrackCharge(charge), &*field);

  //
  // initial error
  //
  AlgebraicSymMatrix55 C = ROOT::Math::SMatrixIdentity();
  float sin2th = sqr(sin(theta));
  float minC00 = 1.0;
  C[0][0] = std::max(sin2th/sqr(ptMin), minC00);
  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy
  C[3][3] = transverseErr;
  C[4][4] = zErr*sin2th + transverseErr*(1-sin2th);
  CurvilinearTrajectoryError initialError(C);

  FreeTrajectoryState fts(initialKine, initialError);

  // get propagator
  edm::ESHandle<Propagator>  propagator;
  es.get<TrackingComponentsRecord>().get(thePropagatorLabel, propagator);

  // get updator
  KFUpdator  updator;

  // Now update initial state track using information from hits.
  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;

  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit];
    if (iHit==0) updatedState = propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface());
    TrajectoryStateOnSurface state = propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return 0;
//    TransientTrackingRecHit::RecHitPointer recHit = (ttrhb->build(hit))->clone(state);
    TransientTrackingRecHit::RecHitPointer recHit =  (ttrhb->build(hit));
    updatedState =  updator.update(state, *recHit);
    if (!updatedState.isValid()) return 0;
  }


  // get propagator
  edm::ESHandle<Propagator>  opropagator;
  es.get<TrackingComponentsRecord>().get(thePropagatorOppositeLabel, opropagator);
  updatedState.rescaleError(100000.);
  for ( int iHit = 2; iHit >= 0; --iHit) {
    hit = hits[iHit];
    TrajectoryStateOnSurface state = 
       opropagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return 0;

//    TransientTrackingRecHit::RecHitPointer recHit = (ttrhb->build(hit))->clone(state);
    TransientTrackingRecHit::RecHitPointer recHit = ttrhb->build(hit);
    updatedState =  updator.update(state, *recHit);
    if (!updatedState.isValid()) return 0;
  }

  TrajectoryStateOnSurface  impactPointState =  
      TransverseImpactPointExtrapolator(&*field).extrapolate( updatedState, vertexPos);
  if (!impactPointState.isValid()) return 0;



  int ndof = 2*hits.size()-5;
  GlobalPoint vv = impactPointState.globalPosition();
  math::XYZPoint  pos( vv.x(), vv.y(), vv.z() );
  GlobalVector pp = impactPointState.globalMomentum();
  math::XYZVector mom( pp.x(), pp.y(), pp.z() );

  float chi2 = 0.;
  reco::Track * track = new reco::Track( chi2, ndof, pos, mom,
        impactPointState.charge(), impactPointState.curvilinearError());

//  std::cout <<"TRACK CREATED" << std::endl;
  return track;
}
