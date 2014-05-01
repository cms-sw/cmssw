///
/// \class l1t::CaloStage2JetSumAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetSumAlgorithmFirmware.h"



l1t::CaloStage2JetSumAlgorithmFirmwareImp1::CaloStage2JetSumAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::CaloStage2JetSumAlgorithmFirmwareImp1::~CaloStage2JetSumAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & jets,
							      std::vector<l1t::EtSum> & etsums) {
   math::XYZTLorentzVector p4;
   int32_t totalHt(0);
   float phiMissingHt;
   int32_t missingHt(0);
   int32_t coefficientX;
   int32_t coefficientY;
   int32_t ptJet;
   double htXComponent(0.);
   double htYComponent(0.);
   int32_t intPhiMissingHt(0);
   const float pi = acos(-1.); 
   float jetPhi;

   for(size_t jetNr=0;jetNr<jets.size();jetNr++)
   {
      if (abs(jets[jetNr].hwEta()) > 28) continue;
      if (jets[jetNr].hwPt() < 80) continue;

      ptJet = (jets[jetNr]).hwPt();
      jetPhi=((jets[jetNr]).hwPhi()*5.0-2.5)*pi/180.;
      coefficientX = int32_t(511.*cos(jetPhi));
      coefficientY = int32_t(511.*sin(jetPhi));

      totalHt += ptJet;
      htXComponent += coefficientX*ptJet;  
      htYComponent += coefficientY*ptJet;  
   }
   htYComponent /= 511.;
   htXComponent /= 511.;  

   phiMissingHt = atan2(htYComponent,htXComponent)+pi;
   if (phiMissingHt > pi) phiMissingHt = phiMissingHt - 2*pi;

   double phi_degrees = phiMissingHt *  180.0 /pi;

   if(phi_degrees < 0) {
      intPhiMissingHt= 72 - (int32_t)(fabs(phi_degrees) / 5.0);
   } else {
      intPhiMissingHt= 1 + (int32_t)(phi_degrees / 5.0);
   } 

   double doubmissingHt = htXComponent*htXComponent+htYComponent*htYComponent;
   missingHt = int32_t(sqrt(doubmissingHt));
   missingHt = missingHt & 0xfff;
   totalHt = totalHt & 0xfff;

   l1t::EtSum::EtSumType typeTotalHt = l1t::EtSum::EtSumType::kTotalHt;
   l1t::EtSum::EtSumType typeMissingHt = l1t::EtSum::EtSumType::kMissingHt;

   l1t::EtSum htSumTotalHt(p4,typeTotalHt,totalHt,0,0,0);
   l1t::EtSum htSumMissingHt(p4,typeMissingHt,missingHt,0,intPhiMissingHt,0);

   etsums.push_back(htSumTotalHt);
   etsums.push_back(htSumMissingHt);
}

