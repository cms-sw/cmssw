#ifndef TrackAssociator_TrackDetMatchInfo_h
#define TrackAssociator_TrackDetMatchInfo_h

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

class TrackDetMatchInfo {
 public:
   /// ECAL energy 
   double ecalEnergyFromRecHits();
   double ecalConeEnergyFromRecHits();
   double ecalEnergyFromCaloTowers();
   double ecalConeEnergyFromCaloTowers();
   /// HCAL energy 
   double hcalEnergyFromRecHits();
   double hcalEnergyFromCaloTowers();
   double hcalConeEnergyFromRecHits();
   double hcalConeEnergyFromCaloTowers();
   double hcalBoxEnergyFromRecHits();
   double hcalBoxEnergyFromCaloTowers();
   
   double outerHcalEnergy();
   
   math::XYZPoint trkGlobPosAtEcal;
   std::vector<EcalRecHit> crossedEcalRecHits;
   std::vector<EcalRecHit> coneEcalRecHits;
   
   math::XYZPoint trkGlobPosAtHcal;
   std::vector<CaloTower> crossedTowers;
   std::vector<CaloTower> coneTowers;
   std::vector<CaloTower> boxTowers;
   std::vector<CaloTower> regionTowers;
   std::vector<HBHERecHit> crossedHcalRecHits;
   std::vector<HBHERecHit> coneHcalRecHits;
   std::vector<HBHERecHit> boxHcalRecHits;
   std::vector<HBHERecHit> regionHcalRecHits;
   
   bool isGoodEcal;
   bool isGoodHcal;
   bool isGoodCalo;
   
};

#endif
