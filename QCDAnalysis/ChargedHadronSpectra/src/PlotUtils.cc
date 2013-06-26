#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotUtils.h"

using namespace std;

/*****************************************************************************/
void PlotUtils::printHelix
  (const GlobalPoint& p1, const GlobalPoint& p2,
   const GlobalVector& v2, ofstream& outFile, int charge)
{
  GlobalVector dp = p2 - p1;
  GlobalVector n2(-v2.y(),v2.x(),0.);
  n2 = n2.unit();

  double r = -0.5 * (dp.x()*dp.x() + dp.y()*dp.y()) /
                    (dp.x()*n2.x() + dp.y()*n2.y());
  GlobalPoint c = p2 + r * n2;

  double dphi = sqrt(2 * 0.1 / fabs(r)); // allowed deflection: 0.1 cm

  double phi = acos(( (p1-c).x()*(p2-c).x() +
                      (p1-c).y()*(p2-c).y() )/(r*r));

  if(dp.x()*v2.x() + dp.y()*v2.y() < 0) phi = 2*M_PI - phi;

  int nstep = (int)(phi/dphi) + 1;

  if(nstep > 1)
  {
    dphi = phi / nstep;
    double dz = (p2 - p1).z() / nstep;


    GlobalPoint P0 = p1;
    GlobalPoint P1;

    charge = ((p1 - c).x() * (p2 - c).y() - (p1 - c).y() * (p2 - c).x() > 0 ?
              -1 : 1);
    if(dp.x()*v2.x() + dp.y()*v2.y() < 0) charge = -charge;

    outFile << ", Line[{{"<<P0.x()<<","<<P0.y()<<",("<<P0.z()<<"-zs)*mz}" ;

    for(int i = 0; i - nstep < 0; i++)
    {
      double a = -charge * (i+1)*dphi;
      double z = p1.z() + (i+1)*dz;

      double x = c.x() + cos(a)*(p1 - c).x() - sin(a)*(p1 - c).y();
      double y = c.y() + sin(a)*(p1 - c).x() + cos(a)*(p1 - c).y();

      P1 = GlobalPoint(x,y,z);

      outFile << ", {"<<P1.x()<<","<<P1.y()<<",("<<P1.z()<<"-zs)*mz}";
      P0 = P1;
    }
    outFile << "}]" << std::endl;
  }
  else
  {
    GlobalPoint P0 = p1;
    GlobalPoint P1 = p2;

    outFile << ", Line[{{"<<P0.x()<<","<<P0.y()<<",("<<P0.z()<<"-zs)*mz}"
                  << ", {"<<P1.x()<<","<<P1.y()<<",("<<P1.z()<<"-zs)*mz}}]" << std::endl;
  }
}
