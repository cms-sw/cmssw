#include "RZLine.h"
#include "CommonTools/Statistics/interface/LinearFit.h"

using namespace std;
template <class T> inline T sqr( T t) {return t*t;}

RZLine::RZLine(const std::vector<float> & aR, 
	       const std::vector<float> & aZ, 
	       const std::vector<float> & aErrZ)
  : r(aR), z(aZ), errZ(aErrZ)
{}

RZLine::RZLine(const vector<GlobalPoint> & points, 
	       const vector<GlobalError> & errors, 
	       const vector<bool> isBarrel) : r(points.size()), z(points.size()), errZ(points.size());
{
  int nPoints = points.size();
  for (int i=0; i<nPoints; i++) {
    const GlobalPoint & p = points[i];
    r[i] = p.perp();
    z[i] = p.z();
  }

  float simpleCot = ( z.back()-z.front() )/ (r.back() - r.front() );
  for (int i=0; i<nPoints; i++) {
    errZ[i] = (isBarrel[i]) ? std::sqrt(errors[i].czz()) :  
      std::sqrt(errors[i].rerr(points[i]) ) * simpleCot;
  }
}

void RZLine::fit(float & cotTheta, float & intercept, 
    float &covss, float &covii, float &covsi) const
{
  LinearFit().fit( r,z, r.size(), errZ, cotTheta, intercept, covss, covii, covsi);
}

float RZLine::chi2(float cotTheta, float intercept) const
{
  float chi2 = 0.f;
  int r_size = r.size();  
  for (int i=0; i< r_size; i++) chi2 += sqr( ((z[i]-intercept) - cotTheta*r[i]) / errZ[i]);
  return chi2;
}
