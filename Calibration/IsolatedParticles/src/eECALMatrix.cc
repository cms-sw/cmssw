#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
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
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "Inside eECALmatrix " << 2*ieta+1 << "X" << 2*iphi+1
                << " nXtals " << vdets.size() << std::endl;
   }
#endif

    const EcalRecHitCollection * recHitsEB = (hitsEB.isValid()) ?
      hitsEB.product() : nullptr;
    bool   flag(true);
    double energySum(0.0);
    for (const auto& id : vdets) {
      if (id != DetId(0)) {
	bool ok = true;
	std::vector<EcalRecHitCollection::const_iterator> hits;
        if (id.subdetId()==EcalBarrel) {
          spr::findHit(hitsEB,id,hits,debug);
	  ok  = (sevlv->severityLevel(id,(*recHitsEB)) != EcalSeverityLevel::kWeird);
        } else if (id.subdetId()==EcalEndcap) {
          spr::findHit(hitsEE,id,hits,debug);
        }
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << "Xtal 0x" << std::hex << id() <<std::dec;
#endif
        double ener(0);
	double ethr = (id.subdetId() !=EcalBarrel) ? eeThr : ebThr;
	for (const auto& hit : hits) {
	  double en(0), tt(0);
	  if (id.subdetId()==EcalBarrel) {
	    if (hit != hitsEB->end()) {
	      en = hit->energy();
              tt = hit->time();
            }
	  } else if (id.subdetId()==EcalEndcap) {
	    if (hit != hitsEE->end()) {
	      en = hit->energy();
              tt = hit->time();
            }
	  }
#ifdef EDM_ML_DEBUG
	  if (debug) std::cout << " " << tt << " " << en;
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
    if (debug) std::cout << "energyECAL: energySum = " << energySum 
			 << " flag = " << flag << std::endl;
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
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "Inside eECALmatrix " << 2*ieta+1 << "X" << 2*iphi+1
                << " nXtals " << vdets.size() << std::endl;
   }
#endif

    const EcalRecHitCollection * recHitsEB =  (hitsEB.isValid()) ?
      hitsEB.product() : nullptr;
    bool   flag(true);
    double energySum = 0.0;
    for (const auto & id : vdets) {
      if (id != DetId(0)) {
        double eTower = spr::energyECALTower(id, hitsEB, hitsEE, ttMap, debug);
        bool   ok(true);
        if      (id.subdetId()==EcalBarrel) ok = (eTower > ebThr);
        else if (id.subdetId()==EcalEndcap) ok = (eTower > eeThr);
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << "Crystal 0x" << std::hex << id() 
			     << std::dec << " Flag " << ok;
#endif
        if (ok) {
	  std::vector<EcalRecHitCollection::const_iterator> hits;
	  if (id.subdetId()==EcalBarrel) {
	    spr::findHit(hitsEB,id,hits,debug);
	    ok  = (sevlv->severityLevel(id,(*recHitsEB)) != EcalSeverityLevel::kWeird);
	  } else if (id.subdetId()==EcalEndcap) {
	    spr::findHit(hitsEE,id,hits,debug);
	  }
	  double ener(0);
	  for (const auto& hit : hits) {
	    double en(0), tt(0);
	    if (id.subdetId()==EcalBarrel) {
	      if (hit != hitsEB->end()) {
		en = hit->energy();
		tt = hit->time();
	      }
	    } else if (id.subdetId()==EcalEndcap) {
	      if (hit != hitsEE->end()) {
		en = hit->energy();
		tt = hit->time();
	      }
	    }
#ifdef EDM_ML_DEBUG
	    if (debug) std::cout << " E " << en << " T " << tt;
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
    if (debug) std::cout << "energyECAL: energySum = " << energySum 
			 << " flag = " << flag << std::endl;
#endif
    return std::pair<double,bool>(energySum,flag);
  }

  std::pair<double,bool> eECALmatrix(const HcalDetId& detId, 
				     edm::Handle<EcalRecHitCollection>& hitsEB,
				     edm::Handle<EcalRecHitCollection>& hitsEE,
				     const CaloGeometry* geo, 
				     const CaloTowerConstituentsMap* ctmap,
				     const EcalSeverityLevelAlgo* sevlv,
				     double ebThr, double eeThr, double tMin, 
				     double tMax, bool debug) {

    CaloTowerDetId tower = ctmap->towerOf(detId);
    std::vector<DetId> ids = ctmap->constituentsOf(tower);
#ifdef EDM_ML_DEBUG
    if (debug) {
      std::cout << "eECALmatrix: " << detId << " belongs to " << tower 
		<< " which has " << ids.size() << " constituents" << std::endl;
      for (unsigned int i=0; i<ids.size(); ++i) {
	std::cout << "[" << i << "] " << std::hex << ids[i].rawId() <<std::dec;
	if        (ids[i].det()==DetId::Ecal && ids[i].subdetId()==EcalBarrel){
	  std::cout << " " << EBDetId(ids[i]) << std::endl;
	} else if (ids[i].det()==DetId::Ecal && ids[i].subdetId()==EcalEndcap){
	  std::cout << " " << EEDetId(ids[i]) << std::endl;
	} else if (ids[i].det()==DetId::Ecal && ids[i].subdetId()==EcalPreshower) {
	  std::cout << " " << ESDetId(ids[i]) << std::endl;
	} else if (ids[i].det()==DetId::Hcal) {
	  std::cout << " " << HcalDetId(ids[i]) << std::endl;
	} else {
	  std::cout << std::endl;
	}
      }
    }
#endif

    bool   flag(true);
    double energySum(0);
    const EcalRecHitCollection* recHitsEB = (hitsEB.isValid()) ? 
      hitsEB.product() : nullptr;
    for (const auto& id : ids) {
      if (id.det() == DetId::Ecal) {
	bool ok(true);
	std::vector<EcalRecHitCollection::const_iterator> hits;
        if (id.subdetId()==EcalBarrel) {
          spr::findHit(hitsEB,id,hits,debug);
	  ok  = (sevlv->severityLevel(id,(*recHitsEB)) != EcalSeverityLevel::kWeird);
        } else if (id.subdetId()==EcalEndcap) {
          spr::findHit(hitsEE,id,hits,debug);
        }
#ifdef EDM_ML_DEBUG
        if (debug) std::cout << "Xtal 0x" << std::hex << id() <<std::dec;
#endif
        double ener(0);
	double ethr = (id.subdetId() !=EcalBarrel) ? eeThr : ebThr;
	for (const auto hit : hits) {
	  double en(0), tt(0);
	  if (id.subdetId()==EcalBarrel) {
	    if (hit != hitsEB->end()) {
	      en = hit->energy();
              tt = hit->time();
            }
	  } else if (id.subdetId()==EcalEndcap) {
	    if (hit != hitsEE->end()) {
	      en = hit->energy();
              tt = hit->time();
            }
	  }
#ifdef EDM_ML_DEBUG
	  if (debug) std::cout << " " << tt << " " << en;
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
    if (debug) std::cout << "energyECAL: energySum = " << energySum 
			 << " flag = " << flag << std::endl;
#endif
    return std::pair<double,bool>(energySum,flag);
  }
}
