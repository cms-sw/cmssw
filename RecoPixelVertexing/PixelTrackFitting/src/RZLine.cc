#include "RZLine.h"
#include "CommonTools/Statistics/interface/LinearFit.h"

using namespace std;
template <class T> inline T sqr( T t) {return t*t;}

RZLine::RZLine(const std::vector<float> & aR, 
	       const std::vector<float> & aZ, 
	       const std::vector<float> & aErrZ) :
  storage(3*aR.size()) {
  nPoints = aR.size();
  r = &storage.front();
  z = r+nPoints;
  errZ2 = z+nPoints;
  for (int i=0; i<nPoints; i++) {
    r[i] = aR[i];
    z[i]=aZ[i];
    errZ2[i] = aErrZ[i]*aErrZ[i];
  } 
}

RZLine::RZLine(const vector<GlobalPoint> & points, 
	       const vector<GlobalError> & errors, 
	       const vector<bool>& isBarrel) : 
  storage(3*points.size()) {
  nPoints = points.size();
  r = &storage.front();
  z = r+nPoints;
  errZ2 = z+nPoints;
  for (int i=0; i!=nPoints; ++i) {
    const GlobalPoint & p = points[i];
    r[i] = p.perp();
    z[i] = p.z();
  }

  float simpleCot2 = ( z[nPoints-1]-z[0] )/ (r[nPoints-1] - r[0] );
  simpleCot2 *= simpleCot2;
  for (int i=0; i!=nPoints; ++i) {
    errZ2[i] = (isBarrel[i]) ? errors[i].czz() :  
      errors[i].rerr(points[i])  * simpleCot2;
  }
}

void RZLine::fit(float & cotTheta, float & intercept, 
    float &covss, float &covii, float &covsi) const
{
  linearFit( r, z, nPoints, errZ2, cotTheta, intercept, covss, covii, covsi);
}

float RZLine::chi2(float cotTheta, float intercept) const
{
  float chi2 = 0.f;
  for (int i=0; i!=nPoints; ++i) chi2 += sqr( ((z[i]-intercept) - cotTheta*r[i]) ) / errZ2[i];
  return chi2;
}
