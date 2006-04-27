#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"

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

#include "DataFormats/TrackReco/interface/HelixParameters.h"

#include "ConformalMappingFit.h"
#include "LinearFit.h"
template <class T> T sqr( T t) {return t*t;}

PixelFitterByConformalMappingAndLine::PixelFitterByConformalMappingAndLine()
{
//  std::cout << " **** HERE PixelFitterByConformalMappingAndLine CTOR"<< std::endl;
}


const reco::Track* PixelFitterByConformalMappingAndLine::run(
    const edm::EventSetup& es,
    const std::vector<const TrackingRecHit * > & hits,
    const TrackingRegion & region) const
{
  int nhits = hits.size();
  if (nhits < 3) return 0;

  vector<float> z,r, errZ;
  typedef ConformalMappingFit::PointXY PointXY;
  vector<PointXY> xy;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

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
  float cotTheta, intercept, covss, covii, covsi;
  LinearFit().fit( r,z, nhits, errZ, cotTheta, intercept, covss, covii, covsi);



//
// construct track, move elsewhere FIXME!!!!
//
  int charge = parabola.charge();

//
// momentum
//
  Measurement1D curv = parabola.curvature();
  float invPt = PixelRecoUtilities::inversePt( curv.value(), es);
  float valPt =  (invPt > 1.e-4) ? 1./invPt : 1.e4;
  float errPt =PixelRecoUtilities::inversePt(curv.error(), es) * sqr(valPt);
  Measurement1D pt (valPt,errPt);
  cout << " reconstructed momentum: " << pt.value() << endl;
  Measurement1D phi = parabola.directionPhi();
  Measurement1D tip = parabola.impactParameter();

  //
  //momentum
  //
  math::XYZVector mom( valPt*cos( phi.value()),
                       valPt*sin( phi.value()),
                       valPt*cotTheta);

  //
  // point of the closest approax to Beam line
  //
  cout << "TIP value: " <<  tip.value() << endl;
  math::XYZPoint  vtx(  tip.value()*cos( phi.value()),
                        tip.value()*sin( phi.value()),
                        intercept);
  cout <<"vertex: " << vtx << endl;

  // temporary fix!
  vtx = math::XYZPoint(0.,0.,vtx.z());
  //
  //errors (dummy)
  //
  math::Error<6>::type cov; //FIXME - feel

  cout <<" momentum: " << mom << endl;
//  return new reco::Track();

  return new reco::Track( 0.,         // chi2
                          2*nhits-5,  // dof
                          nhits,      // foundHits
                          0,
                          0,          //lost hits
                          charge,
                          vtx,
                          mom,
                          cov);
}


