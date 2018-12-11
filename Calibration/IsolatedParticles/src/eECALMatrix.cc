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

  std::pair<double,bool> energyECAL(const DetId& id,
				    edm::Handle<EcalRecHitCollection>& hitsEC,
				    const EcalSeverityLevelAlgo* sevlv,
				    bool testSpike, double tMin, double tMax,
				    bool debug) {
    std::vector<EcalRecHitCollection::const_iterator> hits;
    spr::findHit(hitsEC,id,hits,debug);
#ifdef EDM_ML_DEBUG
    if (debug) std::cout << "Xtal 0x" << std::hex << id() <<std::dec;
#endif
    const EcalRecHitCollection* recHitsEC = (hitsEC.isValid()) ? hitsEC.product() : nullptr;
    bool flag = (!testSpike) ? true :
      (sevlv->severityLevel(id,(*recHitsEC)) != EcalSeverityLevel::kWeird);
    double ener(0);
    for (const auto& hit : hits) {
      double en(0), tt(0);
      if (hit != hitsEC->end()) {
	en = hit->energy();
	tt = hit->time();
      }
#ifdef EDM_ML_DEBUG
      if (debug) std::cout << " " << tt << " " << en;
#endif
      if (tt > tMin && tt < tMax) ener += en;
    }
#ifdef EDM_ML_DEBUG
    if (!flag && debug) std::cout << " detected to be a spike";
    if (debug) std::cout << std::endl;
#endif
    return std::pair<double,bool>(ener,flag);
  }
  
  std::pair<double,bool> energyECAL(const std::vector<DetId>& vdets, 
				    edm::Handle<EcalRecHitCollection>& hitsEC,
				    const EcalSeverityLevelAlgo* sevlv,
				    bool noThrCut, bool testSpike, double eThr,
				    double tMin, double tMax, bool debug) {
    
    bool   flag(true);
    double energySum(0.0);
    for (const auto& id : vdets) {
      if (id != DetId(0)) {
	std::pair<double,bool> ecalEn = spr::energyECAL(id,hitsEC,sevlv,
							testSpike,tMin,tMax,
							debug);
	if (!ecalEn.second) flag = false;
	if ((ecalEn.first>eThr) || noThrCut) energySum += ecalEn.first;
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
    bool               flag(true);
    for (const auto& id : vdets) {
      if        ((id.det() == DetId::Ecal) && (id.subdetId() == EcalBarrel)) {
	if (sevlv->severityLevel(id,(*hitsEB)) == EcalSeverityLevel::kWeird)
	  flag = false;
      } else if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalEndcap)) {
	if (sevlv->severityLevel(id,(*hitsEE)) == EcalSeverityLevel::kWeird)
	  flag = false;
      }
    }
    return std::pair<double,bool>(spr::energyECAL(vdets,hitsEB,hitsEE,ebThr,eeThr,tMin,tMax,debug),flag);
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

    bool   flag(true);
    double energySum = 0.0;
    for (const auto & id : vdets) {
      if ((id != DetId(0)) && (id.det() == DetId::Ecal) &&
	  ((id.subdetId()==EcalBarrel) || (id.subdetId()==EcalEndcap))) {
        double eTower = spr::energyECALTower(id, hitsEB, hitsEE, ttMap, debug);
        bool   ok     =  (id.subdetId()==EcalBarrel) ?  (eTower > ebThr) : (eTower > eeThr);
#ifdef EDM_ML_DEBUG
        if (debug && (!ok)) std::cout << "Crystal 0x" << std::hex << id() 
				      << std::dec << " Flag " << ok <<std::endl;
#endif
        if (ok) {
	  std::pair<double,bool> ecalEn = (id.subdetId()==EcalBarrel) ?
	    spr::energyECAL(id,hitsEB,sevlv,true,tMin,tMax,debug) :
	    spr::energyECAL(id,hitsEE,sevlv,false,tMin,tMax,debug);
	  if (!ecalEn.second) flag = false;
	  energySum += ecalEn.first;
	}
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
    std::vector<DetId> idEBEE;
    bool               flag(true);
    for (const auto& id : ids) {
      if        ((id.det() == DetId::Ecal) && (id.subdetId() == EcalBarrel)) {
	idEBEE.emplace_back(id);
	if (sevlv->severityLevel(id,(*hitsEB)) == EcalSeverityLevel::kWeird)
	  flag = false;
      } else if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalEndcap)) {
	idEBEE.emplace_back(id);
	if (sevlv->severityLevel(id,(*hitsEE)) == EcalSeverityLevel::kWeird)
	  flag = false;
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "eECALmatrix: with " << idEBEE.size() << " EB+EE hits and "
		<< "spike flag " << flag << std::endl;
#endif
    double etot = (!idEBEE.empty()) ?
      spr::energyECAL(idEBEE,hitsEB,hitsEE,ebThr,eeThr,tMin,tMax,debug) : 0;
    return std::pair<double,bool>(etot,flag);
  }
}
