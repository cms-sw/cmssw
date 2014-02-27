#ifndef BINS_H
#define BINS_H
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

  static bool getBin(std::vector<int>& bins, double value, int& low, int& up);
  static void getBins_int(std::vector<int>& bins, int nBins, double xmin, double xmax, bool plotLog=true);
};



#endif
