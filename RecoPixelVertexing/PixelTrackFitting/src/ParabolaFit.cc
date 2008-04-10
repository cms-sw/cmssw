#include "ParabolaFit.h"
using namespace std; 
template <class T> T sqr( T t) {return t*t;}

void ParabolaFit::addPoint(double x, double y, double w)
{
  hasValues = false;
  hasErrors = false;
  Point p = {x,y,w};
  points.push_back(p);
}

const ParabolaFit::Result & ParabolaFit::result( bool doErrors ) const
{
  if (hasErrors)  return theResult;
  if (hasValues && !doErrors) return theResult;

  double F0,  F1,  F2,  F3,  F4,  F0y,  F1y,  F2y;
         F0 = F1 = F2 = F3 = F4 = F0y = F1y = F2y = 0.;

  typedef vector<Point>::const_iterator IT;
  for (IT ip = points.begin(); ip != points.end(); ip++) {
 
    double pow;
    double x = ip->x;
    double y = ip->y;
    double w = ip->w;

                F0 += w;         F0y += w*y;   
                F1 += w*x;       F1y += w*x*y;
    pow = x*x;  F2 += w*pow;     F2y += w*pow*y;    // x^2
    pow *= x;   F3 += w*pow;                        // x^3
    pow *= x;   F4 += w*pow;                        // x^4
  }

  Column cA = { F0, F1, F2 };
  Column cB = { F1, F2, F3 };
  Column cC = { F2, F3, F4 };
  Column cY = { F0y, F1y, F2y };

  double det0 = det(cA, cB, cC);
  theResult.parA = det(cY, cB, cC) / det0;
  theResult.parB = det(cA, cY, cC) / det0;
  theResult.parC = det(cA, cB, cY) / det0;

  double vAA,  vBB,  vCC,  vAB,  vAC,  vBC;
         vAA = vBB = vCC = vAB = vAC = vBC = 0.;

  hasValues = true;
  if (!doErrors) return theResult;
  
  for (IT ip = points.begin(); ip != points.end(); ip++) {

    double w = ip->w;
    Column cX = {1., ip->x, sqr(ip->x) };

    double dXBC = det(cX, cB, cC);
    double dAXC = det(cA, cX, cC);
    double dABX = det(cA, cB, cX);

    vAA += w * sqr(dXBC);
    vBB += w * sqr(dAXC);
    vCC += w * sqr(dABX);
    vAB += w * dXBC * dAXC;
    vAC += w * dXBC * dABX;
    vBC += w * dAXC * dABX;
  }
   
  theResult.varAA = vAA/sqr(det0);
  theResult.varBB = vBB/sqr(det0);
  theResult.varCC = vCC/sqr(det0);
  theResult.varAB = vAB/sqr(det0);
  theResult.varAC = vAC/sqr(det0);
  theResult.varBC = vBC/sqr(det0);

  hasErrors = true;
  return theResult;
}


double ParabolaFit::chi2() const
{
  if (!hasValues) result( doErr );
  double mychi2 = 0.;
  for ( vector<Point>::const_iterator 
      ip = points.begin(); ip != points.end(); ip++) {
     mychi2 += ip->w * sqr(ip->y - fun(ip->x)); 
  }
  return mychi2;
}

double ParabolaFit::fun(double x) const
{
   return theResult.parA + theResult.parB*x + theResult.parC*x*x;
}

int ParabolaFit::dof() const
{
  int n = points.size();
  return (n > 3) ? n-3 : 0;
}

double ParabolaFit::det(
    const Column & c1, const Column & c2, const Column & c3) const
{
  return   c1.r1 * c2.r2 * c3.r3
         + c2.r1 * c3.r2 * c1.r3
         + c3.r1 * c1.r2 * c2.r3
         - c1.r3 * c2.r2 * c3.r1
         - c2.r3 * c3.r2 * c1.r1
         - c3.r3 * c1.r2 * c2.r1;
}
