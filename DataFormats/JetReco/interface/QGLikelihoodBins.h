#ifndef QGLIKELIHOOD_BINS_H
#define QGLIKELIHOOD_BINS_H
#include <vector>

class Bins{
  public:
  const static int nRhoBins=40;
  const static int nPtBins=20;
  const static int Pt0=20;
  const static int Pt1=2000;
  const static int Rho0=0;
  const static int Rho1=40;
  const static int PtLastExtend=4000;

  static void getBins_int(std::vector<int>& bins, int nBins, double xmin, double xmax, bool plotLog=true);
  static int getBinNumber(std::vector<int>& bins, double value);
};



#endif
