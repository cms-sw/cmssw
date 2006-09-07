#include "RZLine.h"
#include "CommonTools/Statistics/interface/LinearFit.h"

using namespace std;
template <class T> T sqr( T t) {return t*t;}

RZLine::RZLine(const vector<GlobalPoint> & points, const vector<GlobalError> & errors, 
         const vector<bool> isBarrel)
{
  int nPoints = points.size();
  for (int i=0; i<nPoints; i++) {
    const GlobalPoint & p = points[i];
    r.push_back(p.perp());
    z.push_back(p.z());
//    cout <<"r,z: "<<r.back()<<" "<<z.back()<<endl;
  }
  float simpleCot = ( z.back()-z.front() )/ (r.back() - r.front() );
  for (int i=0; i<nPoints; i++) {
    if (isBarrel[i]) {
      errZ.push_back( sqrt(errors[i].czz()) );
//      cout <<"ERROR (BARREL): "<<errZ.back()<<endl;
    } else {
      errZ.push_back( sqrt(errors[i].rerr(points[i]) ) * simpleCot);
//      cout <<"ERROR (ENDCAP): "<<errZ.back()<<endl;
   } 
  }
}

void RZLine::fit(float & cotTheta, float & intercept, 
    float &covss, float &covii, float &covsi) const
{
  LinearFit().fit( r,z, r.size(), errZ, cotTheta, intercept, covss, covii, covsi);
}

float RZLine::chi2(float cotTheta, float intercept) const
{
  float chi2 = 0.;
  int r_size = r.size();  
  for (int i=0; i< r_size; i++) {
//    float chi2add =  sqr( (z[i] - (cotTheta*r[i]+intercept)) / errZ[i] );
//    cout <<"adding to chi2: " <<chi2add << endl;
    chi2 += sqr( ((z[i] - intercept) -  cotTheta*r[i]) / errZ[i] );
  } 
  return chi2;
}
