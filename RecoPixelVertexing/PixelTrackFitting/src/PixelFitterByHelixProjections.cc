#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

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
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


#include "RZLine.h"
#include "CircleFromThreePoints.h"
#include "PixelTrackBuilder.h"

PixelFitterByHelixProjections::PixelFitterByHelixProjections() 
  : theTracker(0), theField(0), theTTRecHitBuilder(0) { }

reco::Track* PixelFitterByHelixProjections::run(
    const edm::EventSetup& es,
    const std::vector<const TrackingRecHit * > & hits,
    const TrackingRegion & region) const
{
  int nhits = hits.size();
  if (nhits < 3) return 0;

  vector<GlobalPoint> points;
  vector<GlobalError> errors;
  vector<bool> isBarrel;
  
  if (!theField || !theTracker || !theTTRecHitBuilder) {

    edm::ESHandle<TrackerGeometry> trackerESH;
    es.get<TrackerDigiGeometryRecord>().get(trackerESH);
    theTracker = trackerESH.product();

    edm::ESHandle<MagneticField> fieldESH;
    es.get<IdealMagneticFieldRecord>().get(fieldESH);
    theField = fieldESH.product();

    edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
    es.get<TransientRecHitRecord>().get("WithoutRefit",ttrhbESH);
    theTTRecHitBuilder = ttrhbESH.product();

  }

  for ( vector<const TrackingRecHit*>::const_iterator ih = hits.begin(); ih != hits.end(); ih++) {

//    const GeomDet * det = theTracker->idToDet( (**ih).geographicalId());
//    GlobalPoint p = det->surface().toGlobal( (**ih).localPosition());

    TransientTrackingRecHit::RecHitPointer recHit = theTTRecHitBuilder->build(*ih);
    points.push_back( recHit->globalPosition());
    errors.push_back( recHit->globalPositionError());
    isBarrel.push_back( recHit->detUnit()->type().isBarrel() );
  }
  
  CircleFromThreePoints circle(points[0],points[1],points[2]);

  int charge = PixelFitterByHelixProjections::charge(points);
  float curvature = circle.curvature();

  float invPt = PixelRecoUtilities::inversePt( circle.curvature(), es);
  float valPt = (invPt > 1.e-4) ? 1./invPt : 1.e4;
  float errPt = 0.055*valPt + 0.017*valPt*valPt;

  CircleFromThreePoints::Vector2D center = circle.center();
  float valTip = charge * (center.mag()-1/curvature);
  float errTip = sqrt(errTip2(valPt, points[3].eta()));

  float valPhi = PixelFitterByHelixProjections::phi(center.x(), center.y(), charge);
  float errPhi = 0.002;

  float valZip = zip(valTip, curvature, points[0],points[1]);
  float errZip = sqrt(errZip2(valPt, points[3].eta()));

  float valCotTheta = PixelFitterByHelixProjections::cotTheta(points);
  float errCotTheta = 0.002;

  RZLine rzLine(points,errors,isBarrel);
  float cottheta, intercept, covss, covii, covsi; 
  rzLine.fit(cottheta, intercept, covss, covii, covsi);
  float chi2 = rzLine.chi2(valCotTheta,valZip);
//  cout <<"simple cot: "<<valCotTheta<<" from fit: "<<cottheta<<" chi2: "<<chi2<< endl;
  

  PixelTrackBuilder builder;
  Measurement1D pt(valPt, errPt);
  Measurement1D phi(valPhi, errPhi);
  Measurement1D cotTheta(valCotTheta, errCotTheta);
  Measurement1D tip(valTip, errTip);
  Measurement1D zip(valZip, errZip);

  return builder.build(pt, phi, cotTheta, tip, zip, chi2, charge, hits, theField);
}

int PixelFitterByHelixProjections::charge(const vector<GlobalPoint> & points) const
{
   GlobalVector v21 = points[1]-points[0];
   GlobalVector v32 = points[2]-points[1];
   float dphi = v32.phi() - v21.phi();
   while (dphi >  M_PI) dphi -= 2*M_PI;
   while (dphi < -M_PI) dphi += 2*M_PI;
   return (dphi > 0) ? -1 : 1;
}

float PixelFitterByHelixProjections::cotTheta(const vector<GlobalPoint> & points) const
{
   GlobalPoint gp1 = points[0];
   GlobalPoint gp2 = points[1];
   float dr = gp2.perp()-gp1.perp();
   float dz = gp2.z()-gp1.z();
   return (fabs(dr) > 1.e-3) ? dz/dr : 0;
}

float PixelFitterByHelixProjections::phi(float xC, float yC, int charge) const{
  float phiC = 0.;

  if (charge>0) phiC = atan2(xC,-yC);
  else phiC = atan2(-xC,yC);

  return phiC;
}

float PixelFitterByHelixProjections::zip(float d0, float curv, 
    const GlobalPoint& pinner, const GlobalPoint& pouter) const
{
//phi = asin(r*rho/2) with asin(x) ~= x+x**3/(2*3)
  float rho3 = curv*curv*curv;
  float r1 = pinner.perp();
  double phi1 = r1*curv/2 + pinner.perp2()*r1*rho3/48.;
  float r2 = pouter.perp();
  double phi2 = r2*curv/2 + pouter.perp2()*r2*rho3/48.;
  double z1 = pinner.z();
  double z2 = pouter.z();

  return z1 - phi1/(phi1-phi2)*(z1-z2);
}


double PixelFitterByHelixProjections::errZip2( float apt, float eta) const 
{
  double ziperr=0;
  float pt = (apt <= 10.) ? apt: 10.;
  double p1=0, p2=0,p3=0,p4=0;
  float feta = fabs(eta);
  if (feta<=0.8){
    p1 = 0.12676e-1;
    p2 = -0.22411e-2;
    p3 = 0.2987e-3;
    p4 = -0.12779e-4;
  } else if (feta <=1.6){
    p1 = 0.24047e-1;
    p2 = -0.66935e-2;
    p3 = 0.88111e-3;
    p4 = -0.38482e-4;
  } else {
    p1 = 0.56084e-1;
    p2 = -0.13960e-1;
    p3 = 0.15744e-2;
    p4 = -0.60757e-4;
  }
  ziperr = p1 + p2*pt + p3*pt*pt +p4*pt*pt*pt;
  return ziperr*ziperr;
}

double PixelFitterByHelixProjections::errTip2(float apt, float eta) const
{
  float pt = (apt <= 10.) ? apt : 10.;
  double p1=0, p2=0;
  float feta = fabs(eta);
  if (feta<=0.8)
    {
      p1=5.9e-3;
      p2=4.7e-3;
    }
  else if (feta <=1.6){
    p1 = 4.9e-3;
    p2 = 7.1e-3;
  }
  else {
    p1 = 6.4e-3;
    p2 = 1.0e-2;
  }
  float err=0;
  if (pt != 0) err = (p1 + p2/pt);
  return err*err;
}


