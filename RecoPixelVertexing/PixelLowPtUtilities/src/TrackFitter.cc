#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackFitter.h"
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
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "CommonTools/Statistics/interface/LinearFit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoPixelVertexing/PixelTrackFitting/src/RZLine.h"
#include "RecoPixelVertexing/PixelTrackFitting/src/CircleFromThreePoints.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"

using namespace std;

/*****************************************************************************/
TrackFitter::TrackFitter
  (const edm::ParameterSet& cfg) :
   theConfig(cfg), theTracker(0), theField(0), theTTRecHitBuilder(0)
{
}

/*****************************************************************************/
reco::Track* TrackFitter::run
  (const edm::EventSetup& es,
   const std::vector<const TrackingRecHit *> & hits,
   const TrackingRegion & region) const
{
  int nhits = hits.size();
  if(nhits <2) return 0;

  vector<GlobalPoint> points;
  vector<GlobalError> errors;
  vector<bool> isBarrel;
  
  if (!theField || !theTracker || !theTTRecHitBuilder)
  {
    edm::ESHandle<TrackerGeometry> trackerESH;
    es.get<TrackerDigiGeometryRecord>().get(trackerESH);
    theTracker = trackerESH.product();

    edm::ESHandle<MagneticField> fieldESH;
    es.get<IdealMagneticFieldRecord>().get(fieldESH);
    theField = fieldESH.product();

    edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
    std::string builderName = theConfig.getParameter<std::string>("TTRHBuilder");
    es.get<TransientRecHitRecord>().get(builderName,ttrhbESH);
    theTTRecHitBuilder = ttrhbESH.product();
  }

  for (vector<const TrackingRecHit*>::const_iterator ih = hits.begin();
                                                     ih!= hits.end(); ih++)
  {
    TransientTrackingRecHit::RecHitPointer recHit =
      theTTRecHitBuilder->build(*ih);

    points.push_back(recHit->globalPosition());
    errors.push_back(recHit->globalPositionError());
    isBarrel.push_back(recHit->detUnit()->type().isBarrel() );
  }

  CircleFromThreePoints circle = (nhits==2) ?
        CircleFromThreePoints(GlobalPoint(0.,0.,0.), points[0], points[1]) :
        CircleFromThreePoints(points[0],points[1],points[2]); 

  int charge = getCharge(points);
  float curvature = circle.curvature();

  // pt
  float invPt = PixelRecoUtilities::inversePt(curvature, es);
  float valPt = (invPt > 1.e-4) ? 1./invPt : 1.e4;
  float errPt = 0.055*valPt + 0.017*valPt*valPt;

  CircleFromThreePoints::Vector2D center = circle.center();

  // tip
  float valTip = charge * (center.mag()-1/curvature);
  // zip
  float valZip = getZip(valTip, curvature, points[0],points[1]);
  // phi
  float valPhi = getPhi(center.x(), center.y(), charge);
  // cot(theta), update zip
  float valCotTheta =
     getCotThetaAndUpdateZip(points[0],points[1], 1/curvature,
                             valPhi,valTip,valZip);

  // errors
  float errTip, errZip;
  getErrTipAndErrZip(valPt, points.back().eta(), errTip,errZip);
  float errPhi      = 0.002;
  float errCotTheta = 0.002;

  float chi2 = 0;
  if(nhits > 2)
  {
    RZLine rzLine(points,errors,isBarrel);
    float      cotTheta, intercept, covss, covii, covsi; 
    rzLine.fit(cotTheta, intercept, covss, covii, covsi);
    chi2 = rzLine.chi2(cotTheta, intercept);
    //FIXME: check which intercept to use!
  }

  // build pixel track
  PixelTrackBuilder builder;

  Measurement1D pt      (valPt,       errPt);
  Measurement1D phi     (valPhi,      errPhi);
  Measurement1D cotTheta(valCotTheta, errCotTheta);
  Measurement1D tip     (valTip,      errTip);
  Measurement1D zip     (valZip,      errZip);

  return builder.build(pt, phi, cotTheta, tip, zip, chi2,
                       charge, hits, theField);
}

/*****************************************************************************/
int TrackFitter::getCharge
  (const vector<GlobalPoint> & points) const
{
   GlobalVector v21 = points[1]-points[0];
   GlobalVector v32 = points[2]-points[1];
   float dphi = v32.phi() - v21.phi();
   while (dphi >  M_PI) dphi -= 2*M_PI;
   while (dphi < -M_PI) dphi += 2*M_PI;
   return (dphi > 0) ? -1 : 1;
}

/*****************************************************************************/
float TrackFitter::getCotThetaAndUpdateZip
  (const GlobalPoint& inner, const GlobalPoint& outer,
   float radius, float phi, float d0, float& zip) const
{
   float chi = phi - M_PI_2;
   GlobalPoint IP(d0*cos(chi), d0*sin(chi),zip);

   float phi1 = 2*asin(0.5*(inner - IP).perp()/radius);
   float phi2 = 2*asin(0.5*(outer - IP).perp()/radius);

   float dr = radius*(phi2 - phi1);
   float dz = outer.z()-inner.z();

   // Recalculate ZIP
   zip = (inner.z()*phi2 - outer.z()*phi1)/(phi2 - phi1);

   return (fabs(dr) > 1.e-3) ? dz/dr : 0;
}

/*****************************************************************************/
float TrackFitter::getPhi
  (float xC, float yC, int charge) const
{
  float phiC;

  if (charge>0) phiC = atan2(xC,-yC);
           else phiC = atan2(-xC,yC);

  return phiC;
}

/*****************************************************************************/
float TrackFitter::getZip
  (float d0, float curv,
   const GlobalPoint& inner, const GlobalPoint& outer) const
{
  // phi = asin(r*rho/2) with asin(x) ~= x+x**3/(2*3)
  float rho3 = curv*curv*curv;

  float r1 = inner.perp();
  double phi1 = r1*curv/2 + inner.perp2()*r1*rho3/48.;

  float r2 = outer.perp();
  double phi2 = r2*curv/2 + outer.perp2()*r2*rho3/48.;

  double z1 = inner.z();
  double z2 = outer.z();

  return z1 - phi1/(phi1-phi2)*(z1-z2);
}

/*****************************************************************************/
void TrackFitter::getErrTipAndErrZip
  (float pt, float eta, float & errTip, float & errZip) const 
{
  float coshEta = cosh(eta);

  { // transverse
    float c_ms = 0.0115; //0.0115;
    float s_le = 0.0095; //0.0123;
    float s_ms2 = c_ms*c_ms / (pt*pt) * coshEta;

    errTip = sqrt(s_le*s_le + s_ms2                  );
  }

  { // z
    float c_ms = 0.0070;
    float s_le = 0.0135;

    errZip = sqrt( (s_le*s_le + c_ms*c_ms/(pt*pt)) * coshEta*coshEta*coshEta);
  }
}

