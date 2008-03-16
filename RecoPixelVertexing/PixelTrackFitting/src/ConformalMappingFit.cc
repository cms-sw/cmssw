#include "ConformalMappingFit.h"
//#include "TrackerReco/TkMSParametrization/interface/PixelRecoUtilities.h"
//#include "CommonDet/DetGeometry/interface/LocalError.h"
//#include "CommonDet/DetGeometry/interface/MeasurementError.h"
//#include "CommonDet/DetGeometry/interface/GlobalError.h"
//#include "CommonDet/BasicDet/interface/Det.h"
//#include "CommonDet/PatternPrimitives/interface/TrajectoryStateOnSurface.h"

//#include "TrackerReco/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace std;
using namespace edm;

template <class T> T sqr( T t) {return t*t;}

ConformalMappingFit::ConformalMappingFit(const vector<PointXY> & hits, const ParameterSet & cfg)
  : theRotation(0), myRotation(false)
{
  vector<float> errRPhi2( hits.size(), 1.);
  init( hits, errRPhi2);
  theFit.skipErrorCalculationByDefault();
  if (cfg.exists("fixImpactParameter")) theFit.fixParC(cfg.getParameter<double>("fixImpactParameter"));
}

/*
ConformalMappingFit::ConformalMappingFit(
      const vector<RecHit>& hits,
      const Rotation * rot,
      const vector<TrajectoryStateOnSurface> * tsos,
      bool useMultScatt, float pt, float zVtx)
  : theRotation(rot), myRotation(false)
{
  vector<PointXY> points;
  vector<float> errRPhi2;

  int hits_size = hits.size();
  for (int i = 0; i < hits_size; i++) {
    GlobalPoint p = (tsos == 0) ?
        hits[i].globalPosition()
      : hits[i].det().toGlobal(hits[i].localPosition( (*tsos)[i]));
    GlobalError err = (tsos == 0) ?
        hits[i].globalPositionError()
      : hits[i].det().toGlobal(hits[i].localPositionError( (*tsos)[i]));
    float err2 = p.perp2() * err.phierr(p);
    if (useMultScatt) {
      MultipleScatteringParametrisation ms(hits[i].layer());
      float cotTheta = (p.z()-zVtx)/p.perp();
      err2 += sqr( ms( pt, cotTheta, PixelRecoPointRZ(0.,zVtx) ) );
    }
    points.push_back( PointXY(p.x(), p.y()) );
    errRPhi2.push_back(err2);
  }
  init (points, errRPhi2, rot);
}
*/

void ConformalMappingFit::init( const vector<PointXY> & hits, 
    const vector<float> & errRPhi2, const Rotation * rot)
{
  typedef ConformalMappingFit::MappedPoint<double> PointUV;
  int hits_size = hits.size();
  for ( int i= 0; i < hits_size; i++) {
    if (!theRotation) findRot( hits[i] );
    PointUV point( hits[i], 1./errRPhi2[i], theRotation); 
    theFit.addPoint( point.u(), point.v(), point.weight());
  }
}

void ConformalMappingFit::findRot(const PointXY & p) 
{
  myRotation = true;
  GlobalVector aX = GlobalVector( p.x(), p.y(), 0.).unit();
  GlobalVector aY( -aX.y(), aX.x(), 0.);
  GlobalVector aZ( 0., 0., 1.);
  theRotation = new Rotation(aX,aY,aZ);
}

ConformalMappingFit::~ConformalMappingFit()
{ if( myRotation) delete theRotation; }

double ConformalMappingFit::phiRot() const
{ return atan2( theRotation->xy(), theRotation->xx() ); }

Measurement1D ConformalMappingFit::curvature() const
{
  double val = fabs( 2. * theFit.parA() );
  double err  = 2.*sqrt(theFit.varAA());
  return Measurement1D(val,err);
}

Measurement1D ConformalMappingFit::directionPhi() const
{
  double val = phiRot() + atan(theFit.parB());
  double err = sqrt(theFit.varBB());
  return Measurement1D(val,err);
}

Measurement1D ConformalMappingFit::impactParameter() const
{
  double val = -theFit.parC(); 
  double err = sqrt(theFit.varCC());
  return Measurement1D(val,err);
}

int ConformalMappingFit::charge() const
{ return (theFit.parA() > 0.) ? -1 : 1; }

