#include "DataFormats/JetReco/interface/QGLikelihoodBins.h"
#include <iostream>
#include <vector>
#include <cmath>


int Bins::getBinNumber(std::vector<int>& bins, double value){
  if(value < bins.front() || value > bins.back()) return -1;
  std::vector<int>::iterator binUp = bins.begin() + 1;
  while(value > *binUp) ++binUp;
  return binUp - bins.begin() - 1;
}

void Bins::getBins_int(std::vector<int>& bins, int nBins, double xmin, double xmax, bool log){
  const double dx = (log ? std::pow((xmax/xmin), (1./(double)nBins)) : ((xmax - xmin)/(double)nBins));
  bins.push_back(xmin);
  double binEdge = xmin;
  for(int i = 1; i < nBins; ++i){
    if(log) binEdge *= dx;
    else{ binEdge = ceil(binEdge); binEdge += dx;}
    bins.push_back(ceil(binEdge));
  }
  bins.push_back(xmax);
}
