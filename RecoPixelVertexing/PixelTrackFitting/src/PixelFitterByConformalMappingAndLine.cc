#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CommonTools/Statistics/interface/LinearFit.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"

#include "ConformalMappingFit.h"
#include "PixelTrackBuilder.h"

using namespace std;

template <class T> T sqr( T t) {return t*t;}


PixelFitterByConformalMappingAndLine::PixelFitterByConformalMappingAndLine(
    const edm::ParameterSet& cfg)
{ }

PixelFitterByConformalMappingAndLine::PixelFitterByConformalMappingAndLine()
{
//  std::cout << " **** HERE PixelFitterByConformalMappingAndLine CTOR"<< std::endl;
}


reco::Track* PixelFitterByConformalMappingAndLine::run(
    const edm::EventSetup& es,
    const std::vector<const TrackingRecHit * > & hits,
    const TrackingRegion & region) const
{
  int nhits = hits.size();
  if (nhits < 3) return 0;

  vector<float> z,r, errZ;
  typedef ConformalMappingFit::PointXY PointXY;
  vector<PointXY> xy;

  //temporary check!!!
//  xy.push_back( PointXY(0.0001, 0.0001) );

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  edm::ESHandle<MagneticField> field;
  es.get<IdealMagneticFieldRecord>().get(field);


  for ( vector<const TrackingRecHit *>::const_iterator
        ih = hits.begin();  ih != hits.end(); ih++) {
    const GeomDet * det = tracker->idToDet( (**ih).geographicalId());
    GlobalPoint p = det->surface().toGlobal( (**ih).localPosition());
    xy.push_back( PointXY(p.x(), p.y()) );
    z.push_back( p.z() );
    r.push_back( p.perp());
  }

  //
  // simple fit to get pt, phi0 used for precise calcul.
  //
  ConformalMappingFit parabola(xy);

  //
  // precalculate theta to correct errors:
  //
//  float simpleCot = ( z.back()-z.front() )/ (r.back() - r.front() );
  for (int i=0; i< nhits; i++) {
    errZ.push_back(0.01); // temporary
// const GeomDet * det = tracker->idToDet( (*hits[i]).geographicalId());
// GlobalError err = det->surface().toGlobal( (*hits[i]).localPositionError());
/*
    GlobalError err = hits[i].globalPositionError();
    r[i] += PixelRecoUtilities::longitudinalBendingCorrection(r[i],simple.pT());
    if (hits[i].layer()->part() == barrel) {
      errZ.push_back( sqrt(err.czz()) );
    } else {
      errZ.push_back( sqrt( err.rerr(hits[i].globalPosition()) )*simpleCot );
    }
*/
  }

  //
  // line fit (R-Z plane)
  //
  float cottheta, intercept, covss, covii, covsi;
  LinearFit().fit( r,z, nhits, errZ, cottheta, intercept, covss, covii, covsi);

//
// parameters for track builder 
//
  Measurement1D curv = parabola.curvature();
  float invPt = PixelRecoUtilities::inversePt( curv.value(), es);
  float valPt =  (invPt > 1.e-4) ? 1./invPt : 1.e4;
  float errPt =PixelRecoUtilities::inversePt(curv.error(), es) * sqr(valPt);
  Measurement1D pt (valPt,errPt);
  cout << " reconstructed momentum: " << pt.value() << endl;
  Measurement1D phi = parabola.directionPhi();
  Measurement1D tip = parabola.impactParameter();
  Measurement1D zip(intercept, sqrt(covii));
  Measurement1D cotTheta(cottheta, sqrt(covss));  
  float chi2 = 0.;
  int charge = parabola.charge();


  PixelTrackBuilder builder;
  return builder.build(pt, phi, cotTheta, tip, zip, chi2, charge, hits,  field.product());
}


