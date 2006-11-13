#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include <math.h>

using namespace std;
using namespace muonisolation;

Cuts::Cuts(const vector<double> & etaBounds, const vector<double> & coneSizes,
      const vector<double> & thresholds)
{
  double minEta = 0.;
  double coneSize = 0;
  double threshold = 0;
  unsigned int nEta = etaBounds.size();
  for (unsigned int i=0; i< nEta; i++) {
    if (i>0) minEta = etaBounds[i-1];
    double maxEta = etaBounds[i];
    if (i < coneSizes.size()) coneSize = coneSizes[i]; 
    if (i < thresholds.size()) threshold = thresholds[i]; 

    CutSpec cut = {muonisolation::Range<double>(minEta,maxEta), coneSize, threshold };
    
    theCuts.push_back( cut ); 
  } 
}

const Cuts::CutSpec & Cuts::operator()(double eta) const
{
  double absEta = fabs(eta);
  unsigned int nCuts = theCuts.size();
  unsigned int idx_eta = nCuts-1;
  for (unsigned int i = 0; i < nCuts; i++) {
    if (absEta < theCuts[i].etaRange.max() ) { idx_eta = i; break; }
  }
  return theCuts[idx_eta];
}
