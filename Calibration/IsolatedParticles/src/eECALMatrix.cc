#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include<iostream>

//#define EDM_ML_DEBUG

namespace spr{

  std::pair<double,bool> eECALmatrix(const DetId& detId, 
				     edm::Handle<EcalRecHitCollection>& hitsEB,
				     edm::Handle<EcalRecHitCollection>& hitsEE,
				     const EcalChannelStatus& chStatus, 
				     const CaloGeometry* geo, 
				     const CaloTopology* caloTopology,
				     const EcalSeverityLevelAlgo* sevlv,
				     int ieta, int iphi, double ebThr, 
				     double eeThr, double tMin, double tMax,
				     bool debug) {

    std::vector<DetId> vdets;
    spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, vdets, debug);

    const EcalRecHitCollection * recHitsEB = 0;
    if (hitsEB.isValid())  recHitsEB = hitsEB.product();
    bool flag = true;
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "Inside eECALmatrix " << 2*ieta+1 << "X" << 2*iphi+1
                << " nXtals " << vdets.size() << std::endl;
   }
#endif
    double energySum = 0.0;
    for (auto & vdet : vdets) {
      if (vdet != DetId(0)) {
	bool ok = true;
	std::vector<EcalRecHitCollection::const_iterator> hit;
        if (vdet.subdetId()==EcalBarrel) {
          spr::findHit(hitsEB,vdet,hit,debug);

	  ok  = (sevlv->severityLevel(vdet, (*recHitsEB)) != EcalSeverityLevel::kWeird);
        } else if (vdet.subdetId()==EcalEndcap) {
          spr::findHit(hitsEE,vdet,hit,debug);
        }
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << "Xtal 0x" <<std::hex << vdets[i1]() <<std::dec;
#endif
        double ener=0, ethr=ebThr;
	if (vdet.subdetId() !=EcalBarrel) ethr = eeThr;
	for (auto & ihit : hit) {
	  double en=0, tt=0;
	  if (vdet.subdetId()==EcalBarrel) {
	    if (ihit != hitsEB->end()) {
	      en = ihit->energy();
              tt = ihit->time();
            }
	  } else if (vdet.subdetId()==EcalEndcap) {
	    if (ihit != hitsEE->end()) {
	      en = ihit->energy();
              tt = ihit->time();
            }
	  }
#ifdef EDM_ML_DEBUG
	  if (debug) std::cout << " " << ihit << " " << en;
#endif
	  if (tt > tMin && tt < tMax) ener += en;
	}
	if (!ok) {
	  flag = false;
#ifdef EDM_ML_DEBUG
	  if (debug) std::cout << " detected to be a spike";
#endif
	}
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << std::endl;
#endif
	if (ener > ethr) energySum += ener;
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "energyECAL: energySum = " << energySum << " flag = " << flag << std::endl;
#endif
    return std::pair<double,bool>(energySum,flag);
  }

  std::pair<double,bool> eECALmatrix(const DetId& detId, 
				     edm::Handle<EcalRecHitCollection>& hitsEB,
				     edm::Handle<EcalRecHitCollection>& hitsEE,
				     const EcalChannelStatus& chStatus,
				     const CaloGeometry* geo, 
				     const CaloTopology* caloTopology, 
				     const EcalSeverityLevelAlgo* sevlv,
				     const EcalTrigTowerConstituentsMap& ttMap,
				     int ieta, int iphi, double ebThr, 
				     double eeThr, double tMin, double tMax,
				     bool debug) {

    std::vector<DetId> vdets;
    spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, vdets, debug);

    const EcalRecHitCollection * recHitsEB = 0;
    if (hitsEB.isValid())  recHitsEB = hitsEB.product();
    bool flag = true;
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "Inside eECALmatrix " << 2*ieta+1 << "X" << 2*iphi+1
                << " nXtals " << vdets.size() << std::endl;
   }
#endif
    double energySum = 0.0;
    for (auto & vdet : vdets) {
      if (vdet != DetId(0)) {
        double eTower = spr::energyECALTower(vdet, hitsEB, hitsEE, ttMap, debug);
        bool ok = true;
        if      (vdet.subdetId()==EcalBarrel) ok = (eTower > ebThr);
        else if (vdet.subdetId()==EcalEndcap) ok = (eTower > eeThr);
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << "Crystal 0x" <<std::hex << vdets[i1]() 
			     <<std::dec << " Flag " << ok;
#endif
        if (ok) {
	  std::vector<EcalRecHitCollection::const_iterator> hit;
	  if (vdet.subdetId()==EcalBarrel) {
	    spr::findHit(hitsEB,vdet,hit,debug);

	    ok  = (sevlv->severityLevel(vdet, (*recHitsEB)) != EcalSeverityLevel::kWeird);
	  } else if (vdet.subdetId()==EcalEndcap) {
	    spr::findHit(hitsEE,vdet,hit,debug);
	  }
	  double ener=0;
	  for (auto & ihit : hit) {
	    double en=0, tt=0;
	    if (vdet.subdetId()==EcalBarrel) {
	      if (ihit != hitsEB->end()) {
		en = ihit->energy();
		tt = ihit->time();
	      }
	    } else if (vdet.subdetId()==EcalEndcap) {
	      if (ihit != hitsEE->end()) {
		en = ihit->energy();
		tt = ihit->time();
	      }
	    }
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << " " << ihit << " E " << en << " T " << tt;
#endif
	    if (tt > tMin && tt < tMax) ener += en;
	  }
	  if (!ok) {
	    flag = false;
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << " detected to be a spike";
#endif
	  }
	  energySum += ener;
	}
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << std::endl;
#endif
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "energyECAL: energySum = " << energySum << " flag = " << flag << std::endl;
#endif
    return std::pair<double,bool>(energySum,flag);
  }
}
