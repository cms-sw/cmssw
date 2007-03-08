#include <algorithm>
#include "Calibration/Tools/interface/TrackDetMatchInfo.h"

double TrackDetMatchInfo::ecalEnergyFromRecHits()
{
   double energy(0);
   for(std::vector<EcalRecHit>::const_iterator hit=crossedEcalRecHits.begin(); hit!=crossedEcalRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::ecalConeEnergyFromRecHits()
{
   double energy(0);
   for(std::vector<EcalRecHit>::const_iterator hit=coneEcalRecHits.begin(); hit!=coneEcalRecHits.end(); hit++) {
     energy += hit->energy();    
//     std::cout<< hit->detid().rawId()<<" "<<hit->energy()<<" "<<energy<<std::endl;
   }
   return energy;
}

double TrackDetMatchInfo::ecalEnergyFromCaloTowers()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++) {
     energy += hit->emEnergy();
   }
   return energy;
}

double TrackDetMatchInfo::ecalConeEnergyFromCaloTowers()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=coneTowers.begin(); hit!=coneTowers.end(); hit++)
     energy += hit->emEnergy();
   return energy;
}

double TrackDetMatchInfo::hcalEnergyFromRecHits()
{
   double energy(0);
   for(std::vector<HBHERecHit>::const_iterator hit=crossedHcalRecHits.begin(); hit!=crossedHcalRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::hcalConeEnergyFromRecHits()
{
   double energy(0);
   for(std::vector<HBHERecHit>::const_iterator hit=coneHcalRecHits.begin(); hit!=coneHcalRecHits.end(); hit++) {
     energy += hit->energy();    
   }
   return energy;
}

double TrackDetMatchInfo::hcalBoxEnergyFromRecHits()
{
   double energy(0);
   for(std::vector<HBHERecHit>::const_iterator hit=boxHcalRecHits.begin(); hit!=boxHcalRecHits.end(); hit++)
     energy += hit->energy();    
   return energy;
}

double TrackDetMatchInfo::hcalEnergyFromCaloTowers()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator tower=crossedTowers.begin(); tower!=crossedTowers.end(); tower++)
     energy += tower->hadEnergy();
   return energy;
}

double TrackDetMatchInfo::hcalConeEnergyFromCaloTowers()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=coneTowers.begin(); hit!=coneTowers.end(); hit++) {
     energy += hit->hadEnergy();
   }
   return energy;
}

double TrackDetMatchInfo::hcalBoxEnergyFromCaloTowers()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=boxTowers.begin(); hit!=boxTowers.end(); hit++)
     energy += hit->hadEnergy();
   return energy;
}

double TrackDetMatchInfo::outerHcalEnergy()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator tower=crossedTowers.begin(); tower!=crossedTowers.end(); tower++)
     energy += tower->outerEnergy();
   return energy;
}
