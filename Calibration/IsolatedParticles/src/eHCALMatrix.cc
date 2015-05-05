#include "Calibration/IsolatedParticles/interface/DebugInfo.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include<algorithm>
#include<iostream>

namespace spr{
  HcalDetId getHotCell(std::vector<HBHERecHitCollection::const_iterator>& hit, bool& includeHO, bool& debug) {

    std::vector<HcalDetId> dets;
    std::vector<double> energies;
    for (unsigned int ihit=0; ihit<hit.size(); ihit++) {
      double energy = getEnergy(hit.at(ihit));
      HcalDetId id0 = hit.at(ihit)->id();
      if ((id0.subdet() != HcalOuter) || includeHO) {
      	HcalDetId id1(id0.subdet(),id0.ieta(),id0.iphi(),1);
        bool found(false);
        for (unsigned int idet=0; idet<dets.size(); ++idet) {
          if (id1 == dets[idet]) {
	    energies[idet] += energy; 
	    found = true;
	    break;
          }
        }
        if (!found) {
          dets.push_back(id1);
          energies.push_back(energy);
        }
      }
    }
    double energyMax(-99.);
    HcalDetId hotCell;
    for (unsigned int ihit=0; ihit<dets.size(); ihit++) {
      if (energies[ihit] > energyMax) {
        energyMax = energies[ihit];
	hotCell   = dets[ihit];
      }
    }
    return hotCell;
  }

  HcalDetId getHotCell(std::vector<std::vector<PCaloHit>::const_iterator>& hit, bool& includeHO, bool& debug) {

    std::vector<HcalDetId> dets;
    std::vector<double> energies;
    for (unsigned int ihit=0; ihit<hit.size(); ihit++) {
      double energy = hit.at(ihit)->energy();
      HcalDetId id0 = hit.at(ihit)->id();
      if ((id0.subdet() != HcalOuter) || includeHO) {
      	HcalDetId id1(id0.subdet(),id0.ieta(),id0.iphi(),1);
        bool found(false);
        for (unsigned int idet=0; idet<dets.size(); ++idet) {
          if (id1 == dets[idet]) {
	    energies[idet] += energy; 
	    found = true;
	    break;
          }
        }
        if (!found) {
          dets.push_back(id1);
          energies.push_back(energy);
        }
      }
    }
    double energyMax(-99.);
    HcalDetId hotCell;
    for (unsigned int ihit=0; ihit<dets.size(); ihit++) {
      if (energies[ihit] > energyMax) {
        energyMax = energies[ihit];
	hotCell   = dets[ihit];
      }
    }
    return hotCell;
  }
}
