#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include<iostream>

CaloTowersCreationAlgo::CaloTowersCreationAlgo()
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
   theHcalTopology(0),
   theGeometry(0),
   theTowerConstituentsMap(0),
   theHOIsUsed(true)
{
}



CaloTowersCreationAlgo::CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double  HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight,
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold,
    bool useHO)

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
   theHOIsUsed(useHO)
{
}


void CaloTowersCreationAlgo::setGeometry(const CaloTowerConstituentsMap* ctt, const HcalTopology* topo, const CaloGeometry* geo) {
  theTowerConstituentsMap=ctt;
  theHcalTopology = topo;
  theGeometry = geo;
  theTowerGeometry=geo->getSubdetectorGeometry(DetId::Calo,CaloTowerDetId::SubdetId);
}

void CaloTowersCreationAlgo::begin() {
  theTowerMap.clear();
}

void CaloTowersCreationAlgo::process(const HBHERecHitCollection& hbhe) { 
  for(HBHERecHitCollection::const_iterator hbheItr = hbhe.begin();
      hbheItr != hbhe.end(); ++hbheItr)
    assignHit(&(*hbheItr));
}

void CaloTowersCreationAlgo::process(const HORecHitCollection& ho) { 
  for(HORecHitCollection::const_iterator hoItr = ho.begin();
      hoItr != ho.end(); ++hoItr)
    assignHit(&(*hoItr));
}  

void CaloTowersCreationAlgo::process(const HFRecHitCollection& hf) { 
  for(HFRecHitCollection::const_iterator hfItr = hf.begin();
      hfItr != hf.end(); ++hfItr)  
    assignHit(&(*hfItr));
}

void CaloTowersCreationAlgo::process(const EcalRecHitCollection& ec) { 
  for(EcalRecHitCollection::const_iterator ecItr = ec.begin();
      ecItr != ec.end(); ++ecItr)  
    assignHit(&(*ecItr));
}

void CaloTowersCreationAlgo::finish(CaloTowerCollection& result) {
  // now copy this map into the final collection
  for(MetaTowerMap::const_iterator mapItr = theTowerMap.begin();
      mapItr != theTowerMap.end(); ++ mapItr) {
    CaloTower ct=convert(mapItr->first,mapItr->second);
    if (ct.energy()>0 && ct.energy()>theEcutTower)
      result.push_back(ct);
  }
  theTowerMap.clear(); // save the memory
}


void CaloTowersCreationAlgo::assignHit(const CaloRecHit * recHit) {
  DetId detId = recHit->detid();
  double threshold, weight;
  getThresholdAndWeight(detId, threshold, weight);

  // SPECIAL handling of tower 28/depth 3 --> half into tower 28 and half into tower 29
  if (detId.det()==DetId::Hcal && 
      HcalDetId(detId).subdet()==HcalEndcap &&
      HcalDetId(detId).depth()==3 &&
      HcalDetId(detId).ietaAbs()==28) {

    CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
    if (towerDetId.null()) return;
    MetaTower & tower28 = find(towerDetId);    
    CaloTowerDetId towerDetId29 = CaloTowerDetId(towerDetId.ieta()+
						 towerDetId.zside(),
						 towerDetId.iphi());
    MetaTower & tower29 = find(towerDetId29);

    double energy = recHit->energy()/2; // NOTE DIVIDE BY 2!!!
    if(energy >= threshold) {
      double e28 = energy * weight;
      double e29 = energy * weight;
      
      tower28.E_had += e28;
      tower28.E += e28;
      tower28.constituents.push_back(detId);    

      tower29.E_had += e29;
      tower29.E += e29;
      tower29.constituents.push_back(detId);    
    }
  } else {
    CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
    if (towerDetId.null()) return;    
    MetaTower & tower = find(towerDetId);


    double energy = recHit->energy();
    if(energy >= threshold) {
      // TODO calculate crystal by crystal
      double e = energy * weight;
      
      DetId::Detector det = detId.det();
      if(det == DetId::Ecal) {
        tower.E_em += e;
        tower.E += e;
      }
      // HCAL
      else {
        HcalDetId hcalDetId(detId);
        if(hcalDetId.subdet() == HcalOuter) {
          tower.E_outer += e;
          if(theHOIsUsed) tower.E += e;
        } 
        // HF calculates EM fraction differently
        else if(hcalDetId.subdet() == HcalForward) {
          if(hcalDetId.depth() == 1) {
            // long fiber, so E_EM = E(Long) - E(Short)
            tower.E_em += e;
          } 
          else {
            // short fiber, EHAD = 2 * E(Short)
            tower.E_em -= e;
            tower.E_had += 2. * e;
          }
          tower.E += e;
        }
        else {
          // HCAL situation normal
          tower.E_had += e;
          tower.E += e;
        }
      }
      tower.constituents.push_back(detId);
    } 
  }
}

CaloTowersCreationAlgo::MetaTower::MetaTower() : E(0),E_em(0),E_had(0),E_outer(0) {
}

CaloTowersCreationAlgo::MetaTower & CaloTowersCreationAlgo::find(const CaloTowerDetId & detId) {
  MetaTowerMap::iterator itr = theTowerMap.find(detId);
  if(itr == theTowerMap.end()) {
    // need to build a new tower
    MetaTower t;

    // store it in the map
    theTowerMap.insert(std::pair<CaloTowerDetId, CaloTowersCreationAlgo::MetaTower>(detId, t));
    itr = theTowerMap.find(detId);
  }
  return itr->second;
}

CaloTower CaloTowersCreationAlgo::convert(const CaloTowerDetId& id, const MetaTower& mt) {
    GlobalPoint p=theTowerGeometry->getGeometry(id)->getPosition();
    double pf=1.0/cosh(p.eta());
    double ecalThres=(id.ietaAbs()<=17)?(theEBSumThreshold):(theEESumThreshold);
    double E=mt.E;
    double E_em=mt.E_em;
    double E_had=mt.E_had;
    double E_outer=mt.E_outer;
    std::vector<DetId> contains=mt.constituents;

    if (id.ietaAbs()<=29 && E_em<ecalThres) { // ignore EM threshold in HF
      E-=E_em;
      E_em=0;
      std::vector<DetId> contains_noecal;
      for (std::vector<DetId>::iterator i=contains.begin(); i!=contains.end(); i++) 
	if (i->det()!=DetId::Ecal) contains_noecal.push_back(*i);
      contains.swap(contains_noecal);
    }
    if (id.ietaAbs()<=29 && E_had<theHcalThreshold) {
      E-=E_had;
      E_had=0;
      E_outer=0;
      std::vector<DetId> contains_nohcal;
      for (std::vector<DetId>::iterator i=contains.begin(); i!=contains.end(); i++) 
	if (i->det()!=DetId::Hcal) contains_nohcal.push_back(*i);
      contains.swap(contains_nohcal);
    }

    CaloTower::Vector v(E*pf,p.eta(),p.phi());
    CaloTower retval(id,v,E_em*pf,E_had*pf,E_outer*pf,
		     -1,-1);
    retval.addConstituents(contains);
    return retval;
} 
  
void CaloTowersCreationAlgo::getThresholdAndWeight(const DetId & detId, double & threshold, double & weight) const {
  DetId::Detector det = detId.det();
  weight=0; // in case the hit is not identified
  if(det == DetId::Ecal) {
    // may or may not be EB.  We'll find out.

    EcalSubdetector subdet = (EcalSubdetector)(detId.subdetId());
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
      if(hcalDetId.ietaAbs() < theHcalTopology->firstHEDoublePhiRing()) {
        threshold = theHESthreshold;
        weight = theHESweight;
      }
      else {
        threshold = theHEDthreshold;
        weight = theHEDweight;
      }
    } else if(subdet == HcalOuter) {
      threshold = theHOthreshold;
      weight = theHOweight;
    } else if(subdet == HcalForward) {
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

