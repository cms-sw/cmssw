#include "Calibration/IsolatedParticles/interface/DebugInfo.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include<algorithm>
#include<iostream>

//#define EDM_ML_DEBUG

namespace spr{
  double eHCALmatrix(const HcalTopology* topology, const DetId& det0, 
		     std::vector<PCaloHit>& hits, int ieta, int iphi, 
		     bool includeHO, double hbThr, double heThr, 
		     double hfThr, double hoThr, double tMin, double tMax, bool
#ifdef EDM_ML_DEBUG
		     debug
#endif
		     ) {

    HcalDetId hcid0(det0.rawId());
    HcalDetId hcid(hcid0.subdet(),hcid0.ieta(),hcid0.iphi(),1);
    DetId det(hcid.rawId());
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "Inside eHCALmatrix " << 2*ieta+1 << "X" << 2*iphi+1 << " Inclusion of HO Flag " << includeHO << std::endl;
#endif
    double energySum(0);
    std::vector<DetId> dets(1,det);
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, includeHO, false);
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "matrixHCALIds::Total number of cells found is " 
		<< vdets.size() << std::endl;
      spr::debugHcalDets(0, vdets);
    }
#endif
    int    khit(0);
    for (unsigned int i=0; i<vdets.size(); i++) {
      std::vector<std::vector<PCaloHit>::const_iterator> hit = spr::findHit(hits, vdets[i]);
      double energy = 0;
      int subdet  = ((HcalDetId)(vdets[i].rawId())).subdet();	
      double eThr = spr::eHCALThreshold(subdet, hbThr, heThr, hfThr, hoThr);
      for (unsigned int ihit=0; ihit<hit.size(); ihit++) {
        if (hit[ihit] != hits.end()) {
          khit++;
#ifdef EDM_ML_DEBUG
	  if (debug) std::cout << "energyHCAL:: Hit " << khit << " " << (HcalDetId)vdets[i] << " E " << hit[ihit]->energy() << " t " << hit[ihit]->time() << std::endl;
#endif
          if (hit[ihit]->time() > tMin && hit[ihit]->time() < tMax) {
            energy += hit[ihit]->energy();
          }
        }
      }
      if (energy>eThr) energySum += energy;
    }

#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "eHCALmatrix::Total energy " << energySum << std::endl;
#endif
    return energySum;
  }

  double eHCALmatrix(const CaloGeometry* geo, const HcalTopology* topology, 
		     const DetId& det0, std::vector<PCaloHit>& hits, int ieta, 
		     int iphi, HcalDetId& hotCell, bool includeHO, bool debug) {

    HcalDetId hcid0(det0.rawId());
    HcalDetId hcid(hcid0.subdet(),hcid0.ieta(),hcid0.iphi(),1);
    DetId det(hcid.rawId());
    std::vector<DetId> dets(1,det);
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, includeHO, debug);
    hotCell = hcid0;

    std::vector<std::vector<PCaloHit>::const_iterator> hitlist;
    for (unsigned int i=0; i<vdets.size(); i++) {
      std::vector<std::vector<PCaloHit>::const_iterator> hit = spr::findHit(hits, vdets[i]);
       hitlist.insert(hitlist.end(), hit.begin(), hit.end());
    }

    double energySum(0);
    for (unsigned int ihit=0; ihit<hitlist.size(); ihit++) 
      energySum += hitlist[ihit]->energy();

    // Get hotCell ID
    dets.clear();
    std::vector<double> energies;
    for (unsigned int ihit=0; ihit<hitlist.size(); ihit++) {
      double energy = hitlist[ihit]->energy();
      HcalDetId id0 = HcalDetId(hitlist[ihit]->id());
      if ((id0.subdet() != HcalOuter) || includeHO) {
      	HcalDetId id1(id0.subdet(),id0.ieta(),id0.iphi(),1);
        bool found(false);
        for (unsigned int idet=0; idet<dets.size(); ++idet) {
          if (id1 == HcalDetId(dets[idet])) {
	    energies[idet] += energy; 
	    found = true;
	    break;
          }
        }
        if (!found) {
          dets.push_back(DetId(id1));
          energies.push_back(energy);
        }
      }
    }
    double energyMax(-99.);
    for (unsigned int ihit=0; ihit<dets.size(); ihit++) {
      if (energies[ihit] > energyMax) {
        energyMax = energies[ihit];
	hotCell   = HcalDetId(dets[ihit]);
      }
    }
    return energySum;
  }

  void energyHCALCell(HcalDetId detId, std::vector<PCaloHit>& hits, 
		      std::vector<std::pair<double,int> >& energyCell, 
		      int maxDepth, double hbThr, double heThr, double hfThr, 
		      double hoThr, double tMin, double tMax, bool 
#ifdef EDM_ML_DEBUG
		      debug
#endif
		      ) {

    energyCell.clear();
    int    subdet  = detId.subdet();	
    double eThr    = spr::eHCALThreshold(subdet, hbThr, heThr, hfThr, hoThr);
    bool   hbhe    = (detId.ietaAbs() == 16);
    int    depthHE = (maxDepth <= 6) ? 3 : 4;
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "energyHCALCell: input ID " << detId << " MaxDepth " << maxDepth << " Threshold (E) " << eThr << " (T) " << tMin << ":" << tMax << std::endl;
#endif
    for (int i=0; i<maxDepth; i++) {
      HcalSubdetector subdet0 = (hbhe) ? ((i+1 >= depthHE) ? HcalEndcap : HcalBarrel) : detId.subdet();
      HcalDetId hcid(subdet0,detId.ieta(),detId.iphi(),i+1);
      DetId det(hcid.rawId());
      std::vector<std::vector<PCaloHit>::const_iterator> hit = spr::findHit(hits, det);
      double energy(0);
      for (unsigned int ihit=0; ihit<hit.size(); ++ihit) {
        if (hit[ihit]->time() > tMin && hit[ihit]->time() < tMax) 
          energy += hit[ihit]->energy();
#ifdef EDM_ML_DEBUG
        if (debug)
          std::cout << "energyHCALCell:: Hit[" << ihit << "] " << hcid << " E " << hit[ihit]->energy() << " t " << hit[ihit]->time() << std::endl;
#endif
      }
#ifdef EDM_ML_DEBUG
      if (debug)
	std::cout << "energyHCALCell:: Cell " << hcid << " E " << energy << " from " << hit.size() << " threshold " << eThr << std::endl;
#endif
      if (energy>eThr && !hit.empty()) {
        energyCell.push_back(std::pair<double,int>(energy,i+1));
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "energyHCALCell:: " << energyCell.size() << " entries from "
                << maxDepth << " depths:";
      for (unsigned int i=0; i<energyCell.size(); ++i) {
        std::cout << " [" << i << "] (" << energyCell[i].first << ":"
                  << energyCell[i].second << ")";
      }
      std::cout << std::endl;
    }
#endif
  }

  HcalDetId getHotCell(std::vector<HBHERecHitCollection::const_iterator>& hit, bool includeHO, bool useRaw, bool) {

    std::vector<HcalDetId> dets;
    std::vector<double> energies;
    for (unsigned int ihit=0; ihit<hit.size(); ihit++) {
      double energy = getRawEnergy(hit.at(ihit), useRaw);
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

  HcalDetId getHotCell(std::vector<std::vector<PCaloHit>::const_iterator>& hit, bool includeHO, bool useRaw, bool) {

    std::vector<HcalDetId> dets;
    std::vector<double> energies;
    for (unsigned int ihit=0; ihit<hit.size(); ihit++) {
      double energy = hit.at(ihit)->energy();
      HcalDetId id0 = getRawEnergy(hit.at(ihit),useRaw);
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

  double eHCALThreshold(int subdet, double hbThr, double heThr, double hfThr,
			double hoThr) {
    double eThr = hbThr;
    if      (subdet == (int)(HcalEndcap))  eThr = heThr;
    else if (subdet == (int)(HcalForward)) eThr = hfThr;
    else if (subdet == (int)(HcalOuter))   eThr = hoThr;
    return eThr;
  }
}
