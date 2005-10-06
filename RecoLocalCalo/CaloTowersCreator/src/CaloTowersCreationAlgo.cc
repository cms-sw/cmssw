#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include<iostream>

CaloTowersCreationAlgo::CaloTowersCreationAlgo(const HcalTopology* topo, const CaloGeometry* geo)
 : theEBthreshold(-1000.),
   theEEthreshold(-1000.),
   theHcalThreshold(-1000.),
   theHBthreshold(-1000.),
   theHESthreshold(-1000.),
   theHEDthreshold(-1000.),
   theHOthreshold(-1000.),
   theHF1threshold(-1000.),
   theHF2threshold(-1000.),
   theEBweight(1.),
   theEEweight(1.),
   theHBweight(1.),
   theHESweight(1.),
   theHEDweight(1.),
   theHOweight(1.),
   theHF1weight(1.),
   theHF2weight(1.),
   theEcutTower(-1000.),
   theEBSumThreshold(-1000.),
   theEESumThreshold(-1000.),
   theHcalTopology(topo),
   theGeometry(geo),
   theHOIsUsedByDefault(true)
{
  theTowerTopology=new CaloTowerTopology(); // for now
}



CaloTowersCreationAlgo::CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double  HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight,
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold,
    const HcalTopology* topo, const CaloGeometry* geo, bool useHODefault)

 : theEBthreshold(EBthreshold),
   theEEthreshold(EEthreshold),
   theHcalThreshold(HcalThreshold),
   theHBthreshold(HBthreshold),
   theHESthreshold(HESthreshold),
   theHEDthreshold(HEDthreshold),
   theHOthreshold(HOthreshold),
   theHF1threshold(HF1threshold),
   theHF2threshold(HF2threshold),
   theEBweight(EBweight),
   theEEweight(EEweight),
   theHBweight(HBweight),
   theHESweight(HESweight),
   theHEDweight(HEDweight),
   theHOweight(HOweight),
   theHF1weight(HF1weight),
   theHF2weight(HF2weight),
   theEcutTower(EcutTower),
   theEBSumThreshold(EBSumThreshold),
   theEESumThreshold(EESumThreshold),
   theHcalTopology(topo),
   theGeometry(geo),
   theHOIsUsedByDefault(useHODefault)
{
  towerGeometry=geo->getSubdetectorGeometry(DetId::Calo,CaloTowerDetId::SubdetId);
}

void CaloTowersCreationAlgo::create(CaloTowerCollection & result, const HBHERecHitCollection& hbhe, 
                            const HORecHitCollection & ho, const HFRecHitCollection& hf)
{

  theTowerMap.clear();
  
  for(HBHERecHitCollection::const_iterator hbheItr = hbhe.begin();
      hbheItr != hbhe.end(); ++hbheItr)
    assignHit(&(*hbheItr));
       
  
  for(HORecHitCollection::const_iterator hoItr = ho.begin();
      hoItr != ho.end(); ++hoItr)
    assignHit(&(*hoItr));
  

  for(HFRecHitCollection::const_iterator hfItr = hf.begin();
      hfItr != hf.end(); ++hfItr)  
    assignHit(&(*hfItr));

  // now copy this map into the final collection
  for(CaloTowerMap::const_iterator mapItr = theTowerMap.begin();
      mapItr != theTowerMap.end(); ++ mapItr)
    result.push_back(mapItr->second);
    
}


void CaloTowersCreationAlgo::assignHit(const CaloRecHit * recHit) {
DetId detId = recHit->detid();
  CaloTowerDetId towerDetId = theTowerTopology->towerOf(detId);
  CaloTower & tower = find(towerDetId);

  double threshold, weight;
  getThresholdAndWeight(detId, threshold, weight);
  double energy = recHit->energy();
  if(energy >= threshold) {
    // TODO calculate crystal by crystal
    double sintheta = 1. / cosh(tower.eta);
    double eT = energy * sintheta * weight;

    DetId::Detector det = detId.det();
    if(det == DetId::Ecal) {
      tower.eT_em += eT;
    }
    // HCAL
    else {
      HcalDetId hcalDetId(detId);
      if(hcalDetId.subdet() == HcalOuter) {
        tower.eT_outer += eT;
      } 
      else {
        tower.eT_had += eT;
      }
    }
    tower.constituents.push_back(detId);
    tower.eT += eT;
  } 
}


CaloTower & CaloTowersCreationAlgo::find(const CaloTowerDetId & detId) {
  CaloTowerMap::iterator itr = theTowerMap.find(detId);
  if(itr == theTowerMap.end()) {
    // need to build a new tower
    CaloTower t(detId);
    GlobalPoint p=towerGeometry->getGeometry(detId)->getPosition();
    t.eta = p.eta();
    t.phi = p.phi();

    // store it in the map
    theTowerMap.insert(std::pair<CaloTowerDetId, CaloTower>(detId, t));
    itr = theTowerMap.find(detId);
  }
  return itr->second;
}
 
  
void CaloTowersCreationAlgo::getThresholdAndWeight(const DetId & detId, double & threshold, double & weight) const {
  DetId::Detector det = detId.det();
  if(det == DetId::Ecal) {
    // may or may not be EB.  We'll find out.
    EBDetId ebDetId(detId);
    EcalSubdetector subdet = ebDetId.subdet();
    if(subdet == EcalBarrel) {
      threshold = theEBthreshold;
      weight = theEBweight;
    }
    else if(subdet == EcalEndcap) {
      threshold = theEEthreshold;
      weight = theEEweight;
    }
  }
  else if(det == DetId::Hcal) {
    HcalDetId hcalDetId(detId);
    HcalSubdetector subdet = hcalDetId.subdet();
    
    if(subdet == HcalBarrel) {
      threshold = theHBthreshold;
      weight = theHBweight;
    }
    
    else if(subdet == HcalEndcap) {
      // check if it's single or double tower
      if(hcalDetId.ieta() < theHcalTopology->firstHEDoublePhiRing()) {
        threshold = theHESthreshold;
        weight = theHESweight;
      }
      else {
        threshold = theHEDthreshold;
        weight = theHEDweight;
      }
    }
    
    else if(subdet == HcalForward) {
      if(hcalDetId.depth() == 1) {
        threshold = theHF1threshold;
        weight = theHF1weight;
      } else {
        threshold = theHF2threshold;
        weight = theHF2weight;
      }
    }
  }
  else {
    std::cout << "BAD CELL det " << det << std::endl;
  }
}

