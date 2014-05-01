///
/// \class l1t::CaloStage2EtSumAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EtSumAlgorithmFirmware.h"



l1t::CaloStage2EtSumAlgorithmFirmwareImp1::CaloStage2EtSumAlgorithmFirmwareImp1(CaloParams* params) :
   params_(params)
{


}


l1t::CaloStage2EtSumAlgorithmFirmwareImp1::~CaloStage2EtSumAlgorithmFirmwareImp1() {


}

void l1t::CaloStage2EtSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
      std::vector<l1t::EtSum> & etsums) {

   math::XYZTLorentzVector p4;
   int32_t totalEt(0);
   float phiMissingEt;
   int32_t missingEt;
   int32_t coefficientX;
   int32_t coefficientY;
   int32_t ptTower;
   double etXComponent(0.);
   double etYComponent(0.);
   int32_t intPhiMissingEt(0);
   const float pi = acos(-1.); 
   float towerPhi;

   for(size_t towerNr=0;towerNr<towers.size();towerNr++)
   {
      if (abs((towers[towerNr]).hwEta()) > 28) continue;
      ptTower = (towers[towerNr]).hwPt();
      towerPhi=((towers[towerNr]).hwPhi()*5.0-2.5)*pi/180.;
      coefficientX = int32_t(511.*cos(towerPhi));
      coefficientY = int32_t(511.*sin(towerPhi));

      totalEt += ptTower;
      etXComponent += coefficientX*ptTower;  
      etYComponent += coefficientY*ptTower;  
   }
   etYComponent /= 511.;
   etXComponent /= 511.;  

   phiMissingEt = atan2(etYComponent,etXComponent)+pi;
   if (phiMissingEt > pi) phiMissingEt = phiMissingEt - 2*pi;

   double phi_degrees = phiMissingEt *  180.0 /pi;

   if(phi_degrees < 0) {
      intPhiMissingEt= 72 - (int32_t)(fabs(phi_degrees) / 5.0);
   } else {
      intPhiMissingEt= 1 + (int32_t)(phi_degrees / 5.0);
   } 

   double doubmissingEt = etXComponent*etXComponent+etYComponent*etYComponent;
   missingEt = int32_t(sqrt(doubmissingEt));

   missingEt = missingEt & 0xfff;
   totalEt = totalEt & 0xfff;

   l1t::EtSum::EtSumType typeTotalEt = l1t::EtSum::EtSumType::kTotalEt;
   l1t::EtSum::EtSumType typeMissingEt = l1t::EtSum::EtSumType::kMissingEt;

   l1t::EtSum etSumTotalEt(p4,typeTotalEt,totalEt,0,0,0);
   l1t::EtSum etSumMissingEt(p4,typeMissingEt,missingEt,0,intPhiMissingEt,0);

   etsums.push_back(etSumTotalEt);
   etsums.push_back(etSumMissingEt);

}

