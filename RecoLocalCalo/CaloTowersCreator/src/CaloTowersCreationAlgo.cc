#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/HcalMaterials.h"
#include "Math/Interpolator.h"
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
   theEBGrid(std::vector<double>(5,10.)),
   theEBWeights(std::vector<double>(5,1.)),
   theEEGrid(std::vector<double>(5,10.)),
   theEEWeights(std::vector<double>(5,1.)),
   theHBGrid(std::vector<double>(5,10.)),
   theHBWeights(std::vector<double>(5,1.)),
   theHESGrid(std::vector<double>(5,10.)),
   theHESWeights(std::vector<double>(5,1.)),
   theHEDGrid(std::vector<double>(5,10.)),
   theHEDWeights(std::vector<double>(5,1.)),
   theHOGrid(std::vector<double>(5,10.)),
   theHOWeights(std::vector<double>(5,1.)),
   theHF1Grid(std::vector<double>(5,10.)),
   theHF1Weights(std::vector<double>(5,1.)),
   theHF2Grid(std::vector<double>(5,10.)),
   theHF2Weights(std::vector<double>(5,1.)),
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
   theHOIsUsed(true),
   // (for momentum reconstruction algorithm)
   theMomConstrMethod(0),
   theMomEmDepth(0),
   theMomHadDepth(0),
   theMomTotDepth(0)
{
}

CaloTowersCreationAlgo::CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double  HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight,
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold,
    bool useHO,
    // (momentum reconstruction algorithm)
    int momConstrMethod,
    double momEmDepth,
    double momHadDepth,
    double momTotDepth)

 : theEBthreshold(EBthreshold),
   theEEthreshold(EEthreshold),
   theHcalThreshold(HcalThreshold),
   theHBthreshold(HBthreshold),
   theHESthreshold(HESthreshold),
   theHEDthreshold(HEDthreshold),
   theHOthreshold(HOthreshold),
   theHF1threshold(HF1threshold),
   theHF2threshold(HF2threshold),
   theEBGrid(std::vector<double>(5,10.)),
   theEBWeights(std::vector<double>(5,1.)),
   theEEGrid(std::vector<double>(5,10.)),
   theEEWeights(std::vector<double>(5,1.)),
   theHBGrid(std::vector<double>(5,10.)),
   theHBWeights(std::vector<double>(5,1.)),
   theHESGrid(std::vector<double>(5,10.)),
   theHESWeights(std::vector<double>(5,1.)),
   theHEDGrid(std::vector<double>(5,10.)),
   theHEDWeights(std::vector<double>(5,1.)),
   theHOGrid(std::vector<double>(5,10.)),
   theHOWeights(std::vector<double>(5,1.)),
   theHF1Grid(std::vector<double>(5,10.)),
   theHF1Weights(std::vector<double>(5,1.)),
   theHF2Grid(std::vector<double>(5,10.)),
   theHF2Weights(std::vector<double>(5,1.)),
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
   theHOIsUsed(useHO),
   // (momentum reconstruction algorithm)
   theMomConstrMethod(momConstrMethod),
   theMomEmDepth(momEmDepth),
   theMomHadDepth(momHadDepth),
   theMomTotDepth(momTotDepth)
{
}

CaloTowersCreationAlgo::CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double  HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    std::vector<double> EBGrid, std::vector<double> EBWeights,
    std::vector<double> EEGrid, std::vector<double> EEWeights,
    std::vector<double> HBGrid, std::vector<double> HBWeights,
    std::vector<double> HESGrid, std::vector<double> HESWeights,
    std::vector<double> HEDGrid, std::vector<double> HEDWeights,
    std::vector<double> HOGrid, std::vector<double> HOWeights,
    std::vector<double> HF1Grid, std::vector<double> HF1Weights,
    std::vector<double> HF2Grid, std::vector<double> HF2Weights,
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight,
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold,
    bool useHO,
    // (for the momentum construction algorithm)
    int momConstrMethod,
    double momEmDepth,
    double momHadDepth,
    double momTotDepth
    )

 : theEBthreshold(EBthreshold),
   theEEthreshold(EEthreshold),
   theHcalThreshold(HcalThreshold),
   theHBthreshold(HBthreshold),
   theHESthreshold(HESthreshold),
   theHEDthreshold(HEDthreshold),
   theHOthreshold(HOthreshold),
   theHF1threshold(HF1threshold),
   theHF2threshold(HF2threshold),
   theEBGrid(EBGrid),
   theEBWeights(EBWeights),
   theEEGrid(EEGrid),
   theEEWeights(EEWeights),
   theHBGrid(HBGrid),
   theHBWeights(HBWeights),
   theHESGrid(HESGrid),
   theHESWeights(HESWeights),
   theHEDGrid(HEDGrid),
   theHEDWeights(HEDWeights),
   theHOGrid(HOGrid),
   theHOWeights(HOWeights),
   theHF1Grid(HF1Grid),
   theHF1Weights(HF1Weights),
   theHF2Grid(HF2Grid),
   theHF2Weights(HF2Weights),
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
   theHOIsUsed(useHO),
   // (momentum reconstruction algorithm)
   theMomConstrMethod(momConstrMethod),
   theMomEmDepth(momEmDepth),
   theMomHadDepth(momHadDepth),
   theMomTotDepth(momTotDepth)
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
void CaloTowersCreationAlgo::process(const CaloTowerCollection& ctc) {
  for(CaloTowerCollection::const_iterator ctcItr = ctc.begin();
      ctcItr != ctc.end(); ++ctcItr) { 
    rescale(&(*ctcItr));
    }
}



void CaloTowersCreationAlgo::finish(CaloTowerCollection& result) {
  // now copy this map into the final collection
  for(MetaTowerMap::const_iterator mapItr = theTowerMap.begin();
      mapItr != theTowerMap.end(); ++ mapItr) {
    CaloTower ct=convert(mapItr->first,mapItr->second);
    if (ct.constituentsSize()>0 && ct.energy()>theEcutTower) {
      result.push_back(ct);
    }
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
      std::pair<DetId,double> mc(detId,e28);
      tower28.metaConstituents.push_back(mc);

      tower29.E_had += e29;
      tower29.E += e29;
      tower29.metaConstituents.push_back(mc);
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
      std::pair<DetId,double> mc(detId,e);
      tower.metaConstituents.push_back(mc);
    } 
  }
}


// this method is not compatible with the new format 
// for now make a quick compatibility "fix" : WILL NOT WORK CORRECTLY with anything 
// except the default simple p4 assignment!!!
void CaloTowersCreationAlgo::rescale(const CaloTower * ct) {
  double threshold, weight;

  CaloTowerDetId towerDetId = ct->id();
//  if (towerDetId.null()) return;    
  MetaTower & tower = find(towerDetId);

  tower.E_em = 0.;
  tower.E_had = 0.;
  tower.E_outer = 0.;
  for (unsigned int i=0; i<ct->constituentsSize(); i++) {
    DetId detId = ct->constituent(i);
    getThresholdAndWeight(detId, threshold, weight);
    DetId::Detector det = detId.det();
    if(det == DetId::Ecal) {
      tower.E_em = ct->emEnergy()*weight;
    }
    else {
      HcalDetId hcalDetId(detId);
      if(hcalDetId.subdet() == HcalForward) {
        if (hcalDetId.depth()==1) tower.E_em = ct->emEnergy()*weight;
        if (hcalDetId.depth()==2) tower.E_had = ct->hadEnergy()*weight;
      }
      else if(hcalDetId.subdet() == HcalOuter) {
        tower.E_outer = ct->outerEnergy()*weight;
      }
      else {
        tower.E_had = ct->hadEnergy()*weight;
      }
    }
    tower.E = tower.E_had+tower.E_em+tower.E_outer;

    // this is to be complient with the new MetaTower setup
    // used only for the default simple vector assignment
    std::pair<DetId, double> mc(detId, 0);
    tower.metaConstituents.push_back(mc);

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

  // moved to respective method
//  GlobalPoint p=theTowerGeometry->getGeometry(id)->getPosition();
//  double pf=1.0/cosh(p.eta());

    double ecalThres=(id.ietaAbs()<=17)?(theEBSumThreshold):(theEESumThreshold);
    double E=mt.E;
    double E_em=mt.E_em;
    double E_had=mt.E_had;
    double E_outer=mt.E_outer;
    std::vector<std::pair<DetId,double> > metaContains=mt.metaConstituents;

    if (id.ietaAbs()<=29 && E_em<ecalThres) { // ignore EM threshold in HF
      E-=E_em;
      E_em=0;
      std::vector<std::pair<DetId,double> > metaContains_noecal;

    for (std::vector<std::pair<DetId,double> >::iterator i=metaContains.begin(); i!=metaContains.end(); ++i) 
	        if (i->first.det()!=DetId::Ecal) metaContains_noecal.push_back(*i);
      metaContains.swap(metaContains_noecal);
    }

    if (id.ietaAbs()<=29 && E_had<theHcalThreshold) {
      E-=E_had;
      // if (theHOIsUsed)  E-=E_outer; // not subtracted before, think it should be done
      E_had=0;
      E_outer=0;
      std::vector<std::pair<DetId,double> > metaContains_nohcal;

      for (std::vector<std::pair<DetId,double> >::iterator i=metaContains.begin(); i!=metaContains.end(); ++i) 
        if (i->first.det()!=DetId::Hcal) metaContains_nohcal.push_back(*i);
      metaContains.swap(metaContains_nohcal);
    }

    // create CaloTower using the selected algorithm

    GlobalPoint emPoint, hadPoint;
    CaloTower::LorentzVector towerP4;

  switch (theMomConstrMethod) {

  case 0 :
    {  // Simple 4-momentum assignment
      GlobalPoint p=theTowerGeometry->getGeometry(id)->getPosition();

      double pf=1.0/cosh(p.eta());
      towerP4 = CaloTower::PolarLorentzVector(E*pf, p.eta(), p.phi(), 0);  // simple momentum assignment, same position
      emPoint  = p;   
      hadPoint = p;
    }  // end case 0
    break;

  case 1 :
    {
      if (id.ietaAbs()<=29) {
        if (E_em>0) {
          emPoint   = emShwrPos(metaContains, theMomEmDepth, E_em);
          double emPf = 1.0/cosh(emPoint.eta());
          towerP4 += CaloTower::PolarLorentzVector(E_em*emPf, emPoint.eta(), emPoint.phi(), 0); 
        }
        if (E_had>0) {
          double E_had_tot = (theHOIsUsed)? E_had+E_outer : E_had; 
          hadPoint  = hadShwrPos(metaContains, theMomHadDepth, E_had_tot);
          double hadPf = 1.0/cosh(hadPoint.eta());
          towerP4 += CaloTower::PolarLorentzVector(E_had_tot*hadPf, hadPoint.eta(), hadPoint.phi(), 0); 
        }
      }
      else {  // forward detector: use the CaloTower position 
        GlobalPoint p=theTowerGeometry->getGeometry(id)->getPosition();
        double pf=1.0/cosh(p.eta());
        towerP4 = CaloTower::PolarLorentzVector(E*pf, p.eta(), p.phi(), 0);  // simple momentum assignment, same position
        emPoint  = p;   
        hadPoint = p;
      }
    }  // end case 1
    break;

  }  // end of decision on p4 reconstruction method


//    CaloTower::LorentzVector lv = caloTowerMomentum(id, metaContains, E, E_em, E_had, E_outer);

    CaloTower retval(id, E_em, E_had, E_outer, -1, -1, towerP4, emPoint, hadPoint);

    std::vector<DetId> contains;
    for (std::vector<std::pair<DetId,double> >::iterator i=metaContains.begin(); i!=metaContains.end(); ++i) 
        contains.push_back(i->first);

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
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theEBGrid,theEBWeights,ROOT::Math::Interpolation::AKIMA);
        weight = my.Eval(theEBEScale);
      }
    }
    else if(subdet == EcalEndcap) {
      threshold = theEEthreshold;
      weight = theEEweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theEEGrid,theEEWeights,ROOT::Math::Interpolation::AKIMA);
        weight = my.Eval(theEEEScale);
      }
    }
  }
  else if(det == DetId::Hcal) {
    HcalDetId hcalDetId(detId);
    HcalSubdetector subdet = hcalDetId.subdet();
    
    if(subdet == HcalBarrel) {
      threshold = theHBthreshold;
      weight = theHBweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theHBGrid,theHBWeights,ROOT::Math::Interpolation::AKIMA);
        weight = my.Eval(theHBEScale);
      }
    }
    
    else if(subdet == HcalEndcap) {
      // check if it's single or double tower
      if(hcalDetId.ietaAbs() < theHcalTopology->firstHEDoublePhiRing()) {
        threshold = theHESthreshold;
        weight = theHESweight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHESGrid,theHESWeights,ROOT::Math::Interpolation::AKIMA);
          weight = my.Eval(theHESEScale);
        }
      }
      else {
        threshold = theHEDthreshold;
        weight = theHEDweight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHEDGrid,theHEDWeights,ROOT::Math::Interpolation::AKIMA);
          weight = my.Eval(theHEDEScale);
        }
      }
    } else if(subdet == HcalOuter) {
      threshold = theHOthreshold;
      weight = theHOweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theHOGrid,theHOWeights,ROOT::Math::Interpolation::AKIMA);
        weight = my.Eval(theHOEScale);
      }
    } else if(subdet == HcalForward) {
      if(hcalDetId.depth() == 1) {
        threshold = theHF1threshold;
        weight = theHF1weight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHF1Grid,theHF1Weights,ROOT::Math::Interpolation::AKIMA);
          weight = my.Eval(theHF1EScale);
        }
      } else {
        threshold = theHF2threshold;
        weight = theHF2weight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHF2Grid,theHF2Weights,ROOT::Math::Interpolation::AKIMA);
          weight = my.Eval(theHF2EScale);
        }
      }
    }
  }
  else {
    std::cout << "BAD CELL det " << det << std::endl;
  }
}

void CaloTowersCreationAlgo::setEBEScale(double scale){
  if (scale>0.00001) *&theEBEScale = scale;
  else *&theEBEScale = 50.;
}

void CaloTowersCreationAlgo::setEEEScale(double scale){
  if (scale>0.00001) *&theEEEScale = scale;
  else *&theEEEScale = 50.;
}

void CaloTowersCreationAlgo::setHBEScale(double scale){
  if (scale>0.00001) *&theHBEScale = scale;
  else *&theHBEScale = 50.;
}

void CaloTowersCreationAlgo::setHESEScale(double scale){
  if (scale>0.00001) *&theHESEScale = scale;
  else *&theHESEScale = 50.;
}

void CaloTowersCreationAlgo::setHEDEScale(double scale){
  if (scale>0.00001) *&theHEDEScale = scale;
  else *&theHEDEScale = 50.;
}

void CaloTowersCreationAlgo::setHOEScale(double scale){
  if (scale>0.00001) *&theHOEScale = scale;
  else *&theHOEScale = 50.;
}

void CaloTowersCreationAlgo::setHF1EScale(double scale){
  if (scale>0.00001) *&theHF1EScale = scale;
  else *&theHF1EScale = 50.;
}

void CaloTowersCreationAlgo::setHF2EScale(double scale){
  if (scale>0.00001) *&theHF2EScale = scale;
  else *&theHF2EScale = 50.;
}


GlobalPoint CaloTowersCreationAlgo::emCrystalShwrPos(DetId detId, float fracDepth) {
   const CaloCellGeometry* cellGeometry = theGeometry->getGeometry(detId);
   GlobalPoint point = cellGeometry->getPosition();  // face of the cell

   if      (fracDepth<0) fracDepth=0;
   else if (fracDepth>1) fracDepth=1;

   if (fracDepth>0.0) {
     CaloCellGeometry::CornersVec cv = cellGeometry->getCorners();
     GlobalPoint backPoint = GlobalPoint( 0.25*( cv[4].x() + cv[5].x() + cv[6].x() + cv[7].x() ),
                                          0.25*( cv[4].y() + cv[5].y() + cv[6].y() + cv[7].y() ),
                                          0.25*( cv[4].z() + cv[5].z() + cv[6].z() + cv[7].z() ) );
     point += fracDepth * (backPoint-point);
   }

   return point;
}

GlobalPoint CaloTowersCreationAlgo::hadSegmentShwrPos(DetId detId, float fracDepth) {
   const CaloCellGeometry* cellGeometry = theGeometry->getGeometry(detId);
   GlobalPoint point = cellGeometry->getPosition();  // face of the cell

   if      (fracDepth<0) fracDepth=0;
   else if (fracDepth>1) fracDepth=1;

   if (fracDepth>0.0) {
     CaloCellGeometry::CornersVec cv = cellGeometry->getCorners();
     GlobalPoint backPoint = GlobalPoint( 0.25*( cv[4].x() + cv[5].x() + cv[6].x() + cv[7].x() ),
                                          0.25*( cv[4].y() + cv[5].y() + cv[6].y() + cv[7].y() ),
                                          0.25*( cv[4].z() + cv[5].z() + cv[6].z() + cv[7].z() ) );
     point += fracDepth * (backPoint-point);
   }

   return point;
}


GlobalPoint CaloTowersCreationAlgo::hadShwrPos(std::vector<std::pair<DetId,double> >& metaContains,
                                               float fracDepth, double hadE) {

                                                  
  if (hadE<=0) return GlobalPoint(0,0,0);

  double hadX = 0.0;
  double hadY = 0.0;
  double hadZ = 0.0;

  int nConst = 0;

  std::vector<std::pair<DetId,double> >::iterator mc_it = metaContains.begin();
  for (; mc_it!=metaContains.end(); ++mc_it) {
    if (mc_it->first.det() != DetId::Hcal) continue;
    ++nConst;

    GlobalPoint p = hadSegmentShwrPos(mc_it->first.det(), fracDepth);

    // longitudinal segmantation: do not weight by energy,
    // get the geometrical position
    hadX += p.x();
    hadY += p.y();
    hadZ += p.z();
  }

   return GlobalPoint(hadX/nConst, hadY/nConst, hadZ/nConst);
}


GlobalPoint CaloTowersCreationAlgo::emShwrPos(std::vector<std::pair<DetId,double> >& metaContains, 
                                              float fracDepth, double emE) {

  if (emE<=0) return GlobalPoint(0,0,0);

  double emX = 0.0;
  double emY = 0.0;
  double emZ = 0.0;

  std::vector<std::pair<DetId,double> >::iterator mc_it = metaContains.begin();
  for (; mc_it!=metaContains.end(); ++mc_it) {
    if (mc_it->first.det() != DetId::Ecal) continue;
    GlobalPoint p = emCrystalShwrPos(mc_it->first.det(), fracDepth);
    double e = mc_it->second;

    emX += p.x() * e;
    emY += p.y() * e;
    emZ += p.z() * e;
  }

   return GlobalPoint(emX/emE, emY/emE, emZ/emE);
}


