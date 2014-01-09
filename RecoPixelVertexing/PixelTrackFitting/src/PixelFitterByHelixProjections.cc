#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "CommonTools/Statistics/interface/LinearFit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "FWCore/Framework/interface/ESWatcher.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RZLine.h"
#include "CircleFromThreePoints.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackErrorParam.h"
#include "DataFormats/GeometryVector/interface/Pi.h"

using namespace std;

namespace {

  int charge(const std::vector<GlobalPoint> & points) {
    // the cross product will tell me...
    float dir = (points[1].x()-points[0].x())*(points[2].y()-points[1].y())
      - (points[1].y()-points[0].y())*(points[2].x()-points[1].x());
    
    /*
      GlobalVector v21 = points[1]-points[0];
      GlobalVector v32 = points[2]-points[1];
      float dphi = v32.phi() - v21.phi();
      while (dphi >  Geom::fpi()) dphi -=  Geom::ftwoPi();
      while (dphi < -Geom::fpi()) dphi +=  Geom::ftwoPi();
      return (dphi > 0) ? -1 : 1;
    */
    return (dir>0) ? -1 : 1;
  }

  float cotTheta(const GlobalPoint& inner, const GlobalPoint& outer) {
    float dr = outer.perp()-inner.perp();
    float dz = outer.z()-inner.z();
    return (std::abs(dr) > 1.e-3f) ? dz/dr : 0;
  }

  inline float func_phi(float xC, float yC, int charge) {
    return  (charge>0) ? std::atan2(xC,-yC) :  std::atan2(-xC,yC);
  }

  float zip(float d0, float phi_p, float curv, 
	    const GlobalPoint& pinner, const GlobalPoint& pouter) {
    //
    //phi = asin(r*rho/2) with asin(x) ~= x+x**3/(2*3) = x(1+x*x/6);
    //
    
    float phi0 = phi_p - Geom::fhalfPi();
    GlobalPoint pca(d0*std::cos(phi0), d0*std::sin(phi0),0.);
    
    float rho2 = curv*curv;
    float r1s = (pinner-pca).perp2();
    double phi1 = std::sqrt(r1s)*(curv*0.5f)*(1.f+r1s*(rho2/24.f));
    float r2s = (pouter-pca).perp2();
    double phi2 = std::sqrt(r2s)*(curv*0.5f)*(1.f+r2s*(rho2/24.f));
    double z1 = pinner.z();
    double z2 = pouter.z();

    if (fabs(curv)>1.e-5) 
      return z1 - phi1/(phi1-phi2)*(z1-z2);
    else {
      double dr = std::max(std::sqrt(r2s)-std::sqrt(r1s),1.e-5f);
      return z1-std::sqrt(r1s)*(z2-z1)/dr;
    }
  }
}
  
PixelFitterByHelixProjections::PixelFitterByHelixProjections(
   const edm::ParameterSet& cfg) 
 : theConfig(cfg), theTracker(0), theField(0), theTTRecHitBuilder(0) {}

reco::Track* PixelFitterByHelixProjections::run(
    const edm::EventSetup& es,
    const std::vector<const TrackingRecHit * > & hits,
    const TrackingRegion & region) const
{
  int nhits = hits.size();
  if (nhits <2) return 0;

  vector<GlobalPoint> points(nhits);
  vector<GlobalError> errors(nhits);
  vector<bool> isBarrel(nhits);
  
  static edm::ESWatcher<TrackerDigiGeometryRecord> watcherTrackerDigiGeometryRecord;
  if (!theTracker || watcherTrackerDigiGeometryRecord.check(es)) {
    edm::ESHandle<TrackerGeometry> trackerESH;
    es.get<TrackerDigiGeometryRecord>().get(trackerESH);
    theTracker = trackerESH.product();
  }

  static edm::ESWatcher<IdealMagneticFieldRecord>  watcherIdealMagneticFieldRecord;
  if (!theField || watcherIdealMagneticFieldRecord.check(es)) {
    edm::ESHandle<MagneticField> fieldESH;
    es.get<IdealMagneticFieldRecord>().get(fieldESH);
    theField = fieldESH.product();
  }

  static edm::ESWatcher<TransientRecHitRecord> watcherTransientRecHitRecord;
  if (!theTTRecHitBuilder || watcherTransientRecHitRecord.check(es)) {
    edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
    std::string builderName = theConfig.getParameter<std::string>("TTRHBuilder");
    es.get<TransientRecHitRecord>().get(builderName,ttrhbESH);
    theTTRecHitBuilder = ttrhbESH.product();
  }


  for ( int i=0; i!=nhits; ++i) {
    TransientTrackingRecHit::RecHitPointer recHit = theTTRecHitBuilder->build(hits[i]);
    points[i]  = GlobalPoint( recHit->globalPosition().x()-region.origin().x(), 
			      recHit->globalPosition().y()-region.origin().y(),
			      recHit->globalPosition().z()-region.origin().z() 
			      );
    errors[i] = recHit->globalPositionError();
    isBarrel[i] = recHit->detUnit()->type().isBarrel();
  }

  CircleFromThreePoints circle = (nhits==2) ?
        CircleFromThreePoints( GlobalPoint(0.,0.,0.), points[0], points[1]) :
        CircleFromThreePoints(points[0],points[1],points[2]); 

  float valPhi, valTip, valPt;

  int iCharge = charge(points);
  float curvature = circle.curvature();

  if ((curvature > 1.e-4)&&
	(likely(theField->inTesla(GlobalPoint(0.,0.,0.)).z()>0.01))) {
    float invPt = PixelRecoUtilities::inversePt( circle.curvature(), es);
    valPt = (invPt > 1.e-4f) ? 1.f/invPt : 1.e4f;
    CircleFromThreePoints::Vector2D center = circle.center();
    valTip = iCharge * (center.mag()-1.f/curvature);
    valPhi = func_phi(center.x(), center.y(), iCharge);
  } 
  else {
    valPt = 1.e4f; 
    GlobalVector direction(points[1]-points[0]);
    valPhi =  direction.phi(); 
    valTip = -points[0].x()*sin(valPhi) + points[0].y()*cos(valPhi); 
  }

  float valCotTheta = cotTheta(points[0],points[1]);
  float valEta = asinh(valCotTheta);
  float valZip = zip(valTip, valPhi, curvature, points[0],points[1]);

  PixelTrackErrorParam param(valEta, valPt);
  float errValPt  = param.errPt();
  float errValCot = param.errCot();
  float errValTip = param.errTip();
  float errValPhi = param.errPhi();
  float errValZip = param.errZip();


  float chi2 = 0;
  if (nhits > 2) {
    RZLine rzLine(points,errors,isBarrel);
    float cottheta, intercept, covss, covii, covsi; 
    rzLine.fit(cottheta, intercept, covss, covii, covsi);
    chi2 = rzLine.chi2(cottheta, intercept);         //FIXME: check which intercept to use!
  }

  PixelTrackBuilder builder;
  Measurement1D pt(valPt, errValPt);
  Measurement1D phi(valPhi, errValPhi);
  Measurement1D cotTheta(valCotTheta, errValCot);
  Measurement1D tip(valTip, errValTip);
  Measurement1D zip(valZip, errValZip);

  return builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits, theField, region.origin() );
}




