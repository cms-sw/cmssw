#include <fstream>
#include <cmath>
#include <vector>
#include "L1Trigger/Phase2L1ParticleFlow/interface/deregionizer/deregionizer_input.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

l1ct::DeregionizerInput::DeregionizerInput(std::vector<float> &regionEtaCenter,
                                           std::vector<float> &regionPhiCenter,
                                           const std::vector<l1ct::OutputRegion> &inputRegions)
    : regionEtaCenter_(regionEtaCenter), regionPhiCenter_(regionPhiCenter) {
  orderedInRegionsPuppis_ = std::vector<std::vector<std::vector<l1ct::PuppiObjEmu> > >(nEtaRegions);
  for (int i = 0, n = nEtaRegions; i < n; i++)
    orderedInRegionsPuppis_[i].resize(nPhiRegions);
  initRegions(inputRegions);
}

// +pi read first & account for 2 small eta regions per phi slice
unsigned int l1ct::DeregionizerInput::orderRegionsInPhi(const float eta, const float phi, const float etaComp) const {
  unsigned int y;
  if (fabs(phi) < 0.35)
    y = (eta < etaComp ? 0 : 1);
  else if (fabs(phi) < 1.05)
    y = (phi > 0 ? (eta < etaComp ? 2 : 3) : (eta < etaComp ? 16 : 17));
  else if (fabs(phi) < 1.75)
    y = (phi > 0 ? (eta < etaComp ? 4 : 5) : (eta < etaComp ? 14 : 15));
  else if (fabs(phi) < 2.45)
    y = (phi > 0 ? (eta < etaComp ? 6 : 7) : (eta < etaComp ? 12 : 13));
  else
    y = (phi > 0 ? (eta < etaComp ? 8 : 9) : (eta < etaComp ? 10 : 11));
  return y;
}

void l1ct::DeregionizerInput::initRegions(const std::vector<l1ct::OutputRegion> &inputRegions) {
  for (int i = 0, n = inputRegions.size(); i < n; i++) {
    unsigned int x, y;
    float eta = regionEtaCenter_[i];
    float phi = regionPhiCenter_[i];

    if (fabs(eta) < 0.5) {
      x = 0;
      y = orderRegionsInPhi(eta, phi, 0.0);
    } else if (fabs(eta) < 1.5) {
      x = (eta < 0.0 ? 1 : 2);
      y = (eta < 0.0 ? orderRegionsInPhi(eta, phi, -1.0) : orderRegionsInPhi(eta, phi, 1.0));
    } else if (fabs(eta) < 2.5) {
      x = (eta < 0.0 ? 3 : 4);
      y = orderRegionsInPhi(eta, phi, 999.0);  // Send all candidates in 3 clks, then wait 3 clks for the barrel
    } else /*if ( fabs(eta) < 3.0 )*/ {
      x = 5;
      y = orderRegionsInPhi(eta, phi, 0.0);  // Send eta<0 in 3 clks, eta>0 in the next 3 clks
    }
    /*else x = 6;*/  // HF

    orderedInRegionsPuppis_[x][y].insert(orderedInRegionsPuppis_[x][y].end(),
                                         inputRegions[i].puppi.begin(),
                                         inputRegions[i].puppi.end());  // For now, merging HF with forward HGCal

    while (!orderedInRegionsPuppis_[x][y].empty() && orderedInRegionsPuppis_[x][y].back().hwPt == 0)
      orderedInRegionsPuppis_[x][y].pop_back();  // Zero suppression
  }
}

void l1ct::DeregionizerInput::orderRegions(int order[nEtaRegions]) {
  std::vector<std::vector<std::vector<l1ct::PuppiObjEmu> > > tmpOrderedInRegionsPuppis;
  for (int i = 0, n = nEtaRegions; i < n; i++)
    tmpOrderedInRegionsPuppis.push_back(orderedInRegionsPuppis_[order[i]]);
  orderedInRegionsPuppis_ = tmpOrderedInRegionsPuppis;

  if (debug_) {
    for (int i = 0, nx = orderedInRegionsPuppis_.size(); i < nx; i++) {
      dbgCout() << "\n";
      dbgCout() << "Eta region index : " << i << "\n";
      for (int j = 0, ny = orderedInRegionsPuppis_[i].size(); j < ny; j++) {
        dbgCout() << " ---> Phi region index : " << j << "\n";
        for (int iPup = 0, nPup = orderedInRegionsPuppis_[i][j].size(); iPup < nPup; iPup++) {
          dbgCout() << "      > puppi[" << iPup << "]"
                    << " pt = " << orderedInRegionsPuppis_[i][j][iPup].hwPt << "\n";
        }
      }
      dbgCout() << " ----------------- "
                << "\n";
    }
    dbgCout() << "Regions ordered!"
              << "\n";
    dbgCout() << "\n";
  }
}
