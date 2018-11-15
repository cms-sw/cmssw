#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ThirdHitPrediction.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/CircleFromThreePoints.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

inline float sqr(float x) { return x*x; }

using namespace std;

/*****************************************************************************/
ThirdHitPrediction::ThirdHitPrediction
  (const TrackingRegion & region, GlobalPoint inner, GlobalPoint outer,
   const edm::EventSetup& es,
   double nSigMultipleScattering, double maxAngleRatio,
   string builderName)
{
 using namespace edm;
 ESHandle<MagneticField> magfield;
 es.get<IdealMagneticFieldRecord>().get(magfield);

 edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
 es.get<TransientRecHitRecord>().get(builderName,ttrhbESH);
 theTTRecHitBuilder = ttrhbESH.product();

 Bz = fabs(magfield->inInverseGeV(GlobalPoint(0,0,0)).z());

 c0 = Global2DVector(region.origin().x(),
                     region.origin().y());

 r0 = region.originRBound();
 rm = region.ptMin() / Bz;

 g1 = inner;
 g2 = outer;

 p1 = Global2DVector(g1.x(), g1.y());
 p2 = Global2DVector(g2.x(), g2.y());

 dif = p1 - p2;

 // Prepare circles of minimal pt (rm) and cylinder of origin (r0)
 keep = true;
 arc_0m = findArcIntersection(findMinimalCircles (rm),
                              findTouchingCircles(r0), keep);

 nSigma   = nSigMultipleScattering;
 maxRatio = maxAngleRatio;
}

/*****************************************************************************/
ThirdHitPrediction::~ThirdHitPrediction()
{
}

/*****************************************************************************/
void ThirdHitPrediction::invertCircle(Global2DVector& c,float& r)
{
  float s = dif.mag2() / ((c - p1).mag2() - sqr(r));

  c = p1 + (c - p1)*s;
  r *= fabsf(s);
}

/*****************************************************************************/
void ThirdHitPrediction::invertPoint(Global2DVector& p)
{
  float s = dif.mag2() / (p - p1).mag2();

  p = p1 + (p - p1)*s;
}

/*****************************************************************************/
pair<float,float> ThirdHitPrediction::findMinimalCircles(float r)
{ 
  pair<float,float> a(0.,0.);

  if(dif.mag2() <  2 * sqr(r))
    a = pair<float,float>( dif.phi(),
                           0.5*acos(1 - 0.5 * dif.mag2()/sqr(r)) );

  return a;
}

/*****************************************************************************/
pair<float,float> ThirdHitPrediction::findTouchingCircles(float r)
{
  Global2DVector c = c0;
  invertCircle(c,r);

  pair<float,float> a(0.,0.);
  a = pair<float,float>( (c - p2).phi(),
                          0.5*acos(1 - 2*sqr(r)/(c - p2).mag2()) );

  return a;
}

/*****************************************************************************/
pair<float,float> ThirdHitPrediction::findArcIntersection
  (pair<float,float> a, pair<float,float> b, bool& keep)
{ 
  // spin closer
  while(b.first < a.first - M_PI) b.first += 2*M_PI;
  while(b.first > a.first + M_PI) b.first -= 2*M_PI;

  float min,max;

  if(a.first - a.second > b.first - b.second)
    min = a.first - a.second;
  else
  { min = b.first - b.second; keep = false; }

  if(a.first + a.second < b.first + b.second)
    max = a.first + a.second;
  else
  { max = b.first + b.second; keep = false; }
  
  pair<float,float> c(0.,0.);

  if(min < max)
  {
    c.first  = 0.5*(max + min);
    c.second = 0.5*(max - min);
  }

  return c;
}

/*****************************************************************************/
void ThirdHitPrediction::fitParabola
  (const float x[3], const float y[3], float par[3])
{ 
  float s2 = sqr(x[0]) * (y[1] - y[2]) +
             sqr(x[1]) * (y[2] - y[0]) +
             sqr(x[2]) * (y[0] - y[1]);
  
  float s1 =     x[0]  * (y[1] - y[2]) +
                 x[1]  * (y[2] - y[0]) +
                 x[2]  * (y[0] - y[1]);

  float s3 = (x[0] - x[1]) * (x[1] - x[2]) * (x[2] - x[0]);
  float s4 = x[0]*x[1]*y[2] * (x[0] - x[1]) +
             x[0]*y[1]*x[2] * (x[2] - x[0]) +
             y[0]*x[1]*x[2] * (x[1] - x[2]);

  par[2] =  s1 / s3; // a2
  par[1] = -s2 / s3; // a1
  par[0] = -s4 / s3; // a0
}

/*****************************************************************************/
void ThirdHitPrediction::findRectangle
  (const float x[3], const float y[3], const float par[3],
   float phi[2],float z[2])
{ 
  // Initial guess
  phi[0] = min(x[0],x[2]); z[0] = min(y[0],y[2]);
  phi[1] = max(x[0],x[2]); z[1] = max(y[0],y[2]);

  // Extremum: position and value
  float xe = -par[1]/(2*par[2]);
  float ye = par[0] - sqr(par[1])/(4*par[2]);

  // Check if extremum is inside the phi range
  if(phi[0] < xe  && xe < phi[1])
  {
    if(ye < z[0]) z[0] = ye;
    if(ye > z[1]) z[1] = ye;
  }
}

/*****************************************************************************/
float ThirdHitPrediction::areaParallelogram
  (const Global2DVector& a, const Global2DVector& b)
{
  return a.x() * b.y() - a.y() * b.x();
}

/*****************************************************************************/
float ThirdHitPrediction::angleRatio
  (const Global2DVector& p3, const Global2DVector& c)
{
  float rad2 = (p1 - c).mag2();

  float a12 = asin(fabsf(areaParallelogram(p1 - c, p2 - c)) / rad2);
  float a23 = asin(fabsf(areaParallelogram(p2 - c, p3 - c)) / rad2);

  return a23/a12;
}

/*****************************************************************************/
void ThirdHitPrediction::spinCloser(float phi[3])
{
  while(phi[1] < phi[0] - M_PI) phi[1] += 2*M_PI;
  while(phi[1] > phi[0] + M_PI) phi[1] -= 2*M_PI;

  while(phi[2] < phi[1] - M_PI) phi[2] += 2*M_PI;
  while(phi[2] > phi[1] + M_PI) phi[2] -= 2*M_PI;
}

/*****************************************************************************/
void ThirdHitPrediction::calculateRangesBarrel
  (float r3, float phi[2],float z[2], bool keep)
{
  pair<float,float> arc_all =
    findArcIntersection(arc_0m, findTouchingCircles(r3), keep);

  if(arc_all.second != 0.)
  {
    Global2DVector c3(0.,0.); // barrel at r3
    invertCircle(c3,r3);      // inverted

    float angle[3];           // prepare angles
    angle[0] = arc_all.first - arc_all.second;
    angle[1] = arc_all.first;
    angle[2] = arc_all.first + arc_all.second;

    float phi3[3], z3[3];
    Global2DVector delta = c3 - p2;

    for(int i=0; i<3; i++)
    {
      Global2DVector vec(cos(angle[i]), sin(angle[i])); // unit vector
      float lambda = delta*vec - sqrt(sqr(delta*vec) - delta*delta + sqr(r3));

      Global2DVector p3 = p2 + lambda * vec;  // inverted third hit 
      invertPoint(p3);                        // third hit
      phi3[i] = p3.phi();                     // phi of third hit

      float ratio;

      if(keep && i==1)
      { // Straight line
        ratio = (p2 - p3).mag() / (p1 - p2).mag();
      }
      else
      { // Circle
        Global2DVector c = p2 - vec * (vec * (p2 - p1)); // inverted antipodal
        invertPoint(c);                                  // antipodal
        c = 0.5*(p1 + c);                                // center

        ratio = angleRatio(p3,c);
      }

      z3[i] = g2.z() + (g2.z() - g1.z()) * ratio;        // z of third hit
    }

    spinCloser(phi3);

    // Parabola on phi - z
    float par[3];
    fitParabola  (phi3,z3, par);
    findRectangle(phi3,z3, par, phi,z);
  }
}

/*****************************************************************************/
void ThirdHitPrediction::calculateRangesForward
  (float z3, float phi[2],float r[2], bool keep)
{
  float angle[3];           // prepare angles
  angle[0] = arc_0m.first - arc_0m.second;
  angle[1] = arc_0m.first;
  angle[2] = arc_0m.first + arc_0m.second;

  float ratio = (z3 - g2.z()) / (g2.z() - g1.z());

  if(0 < ratio  && ratio < maxRatio)
  {
    float phi3[3], r3[3];

    for(int i=0; i<3; i++)
    {
      Global2DVector p3;

      if(keep && i==1)
      { // Straight line
        p3 = p2 + ratio * (p2 - p1);
      }
      else
      { // Circle
        Global2DVector vec(cos(angle[i]), sin(angle[i])); // unit vector
  
        Global2DVector c = p2 - vec * (vec * (p2 - p1));  // inverted antipodal
        invertPoint(c);                                   // antipodal
        c = 0.5*(p1 + c);                                 // center

        float rad2 = (p1 - c).mag2();

        float a12 = asin(areaParallelogram(p1 - c, p2 - c) / rad2);
        float a23 = ratio * a12;

        p3 = c + Global2DVector((p2-c).x()*cos(a23) - (p2-c).y()*sin(a23),
                                (p2-c).x()*sin(a23) + (p2-c).y()*cos(a23));
      }

      phi3[i] = p3.phi();
      r3[i]   = p3.mag();
    }

    spinCloser(phi3);

    // Parabola on phi - z
    float par[3];
    fitParabola  (phi3,r3, par);
    findRectangle(phi3,r3, par, phi,r);
  }
}

/*****************************************************************************/
void ThirdHitPrediction::calculateRanges
  (float rz3, float phi[2],float rz[2])
{
  // Clear
  phi[0] = 0.; rz[0] = 0.;
  phi[1] = 0.; rz[1] = 0.;

  // Calculate
  if(theBarrel) calculateRangesBarrel (rz3, phi,rz, keep);
           else calculateRangesForward(rz3, phi,rz, keep);
}

/*****************************************************************************/
void ThirdHitPrediction::getRanges
  (const DetLayer *layer, float phi[],float rz[])
{
  theLayer = layer;
  
  if (layer) initLayer(layer);

  float phi_inner[2],rz_inner[2];
  calculateRanges(theDetRange.min(), phi_inner,rz_inner);

  float phi_outer[2],rz_outer[2];
  calculateRanges(theDetRange.max(), phi_outer,rz_outer);

  if( (phi_inner[0] == 0. && phi_inner[1] == 0.) ||
      (phi_outer[0] == 0. && phi_outer[1] == 0.) )
  {
    phi[0] = 0.;
    phi[1] = 0.;

     rz[0] = 0.;
     rz[1] = 0.;
  }
  else
  {
    while(phi_outer[0] > phi_inner[0] + M_PI)
    { phi_outer[0] -= 2*M_PI; phi_outer[1] -= 2*M_PI; }

    while(phi_outer[0] < phi_inner[0] - M_PI)
    { phi_outer[0] += 2*M_PI; phi_outer[1] += 2*M_PI; }

    phi[0] = min(phi_inner[0],phi_outer[0]);
    phi[1] = max(phi_inner[1],phi_outer[1]);
  
     rz[0] = min( rz_inner[0], rz_outer[0]);
     rz[1] = max( rz_inner[1], rz_outer[1]);
  }
}

/*****************************************************************************/
void ThirdHitPrediction::getRanges
  (float rz3, float phi[],float rz[])
{
  calculateRanges(rz3, phi,rz);
}

/*****************************************************************************/
bool ThirdHitPrediction::isCompatibleWithMultipleScattering
  (GlobalPoint g3, const vector<const TrackingRecHit*>& h,
   vector<GlobalVector>& globalDirs, const edm::EventSetup& es)
{
  Global2DVector p1(g1.x(),g1.y());
  Global2DVector p2(g2.x(),g2.y());
  Global2DVector p3(g3.x(),g3.y());

  CircleFromThreePoints circle(g1,g2,g3);

  if(circle.curvature() != 0.)
  {
  Global2DVector c (circle.center().x(), circle.center().y());

  float rad2 = (p1 - c).mag2();
  float a12 = asin(fabsf(areaParallelogram(p1 - c, p2 - c)) / rad2);
  float a23 = asin(fabsf(areaParallelogram(p2 - c, p3 - c)) / rad2);

  float slope = (g2.z() - g1.z()) / a12;

  float rz3 = g2.z() + slope * a23;
  float delta_z = g3.z() - rz3;

  // Transform to tt
  vector<TransientTrackingRecHit::RecHitPointer> th;
  for(vector<const TrackingRecHit*>::const_iterator ih = h.begin(); ih!= h.end(); ih++)
    th.push_back(theTTRecHitBuilder->build(*ih));

  float sigma1_le2 = max(th[0]->parametersError()[0][0],
                         th[0]->parametersError()[1][1]);
  float sigma2_le2 = max(th[1]->parametersError()[0][0],
                         th[1]->parametersError()[1][1]);

  float sigma_z2 = (1 + a23/a12)*(1 + a23/a12) * sigma2_le2 +
                   (    a23/a12)*(    a23/a12) * sigma1_le2;

  float cotTheta = slope * circle.curvature(); // == sinhEta
  float coshEta  = sqrt(1 + sqr(cotTheta));    // == 1/sinTheta

  float pt = Bz / circle.curvature();
  float p  = pt * coshEta;

  float m_pi = 0.13957018;
  float beta = p / sqrt(sqr(p) + sqr(m_pi));

  MultipleScatteringParametrisation msp(theLayer,es);
  PixelRecoPointRZ rz2(g2.perp(), g2.z());

  float sigma_z = msp(pt, cotTheta, rz2) / beta;

  // Calculate globalDirs
  float sinTheta =       1. / coshEta;
  float cosTheta = cotTheta * sinTheta;

  int dir;
  if(areaParallelogram(p1 - c, p2 - c) > 0) dir = 1; else dir = -1;

  float curvature = circle.curvature();

  {
   Global2DVector v = (p1 - c)*curvature*dir;
   globalDirs.push_back(GlobalVector(-v.y()*sinTheta,v.x()*sinTheta,cosTheta));
  }

  {
   Global2DVector v = (p2 - c)*curvature*dir;
   globalDirs.push_back(GlobalVector(-v.y()*sinTheta,v.x()*sinTheta,cosTheta));
  }

  {
   Global2DVector v = (p3 - c)*curvature*dir;
   globalDirs.push_back(GlobalVector(-v.y()*sinTheta,v.x()*sinTheta,cosTheta));
  }

  // Multiple scattering
  float sigma_ms  = sigma_z * coshEta;

  // Local error squared
  float sigma_le2 = max(th[2]->parametersError()[0][0],
                        th[2]->parametersError()[1][1]);

  return (delta_z*delta_z / (sigma_ms*sigma_ms + sigma_le2 + sigma_z2)
          < nSigma * nSigma);
  }

  return false;
}

/*****************************************************************************/
void ThirdHitPrediction::initLayer(const DetLayer *layer)
{
  if ( layer->location() == GeomDetEnumerators::barrel) {
    theBarrel = true;
    theForward = false;
    const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*layer);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    theDetRange = Range(radius-halfThickness, radius+halfThickness);
  } else if ( layer->location() == GeomDetEnumerators::endcap) {
    theBarrel= false;
    theForward = true;
    const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*layer);
    float halfThickness  = fl.surface().bounds().thickness()/2;
    float zLayer = fl.position().z() ;
    theDetRange = Range(zLayer-halfThickness, zLayer+halfThickness);
  }
}
