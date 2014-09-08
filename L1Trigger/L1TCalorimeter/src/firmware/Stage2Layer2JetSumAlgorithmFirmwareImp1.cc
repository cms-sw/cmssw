///
/// \class l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetSumAlgorithmFirmware.h"



l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::Stage2Layer2JetSumAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{
  etSumEtThresholdHwEt_ = floor(params_->etSumEtThreshold(2)/params_->jetLsb());
  etSumEtThresholdHwMet_ = floor(params_->etSumEtThreshold(4)/params_->jetLsb());

  etSumEtaMinEt_ = params_->etSumEtaMin(2);
  etSumEtaMaxEt_ = params_->etSumEtaMax(2);
  
  etSumEtaMinMet_ = params_->etSumEtaMin(4);
  etSumEtaMaxMet_ = params_->etSumEtaMax(4);
  

}


l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::~Stage2Layer2JetSumAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & jets,
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
	 ptJet = (jets[jetNr]).hwPt();
      if ((jets[jetNr]).hwEta() > etSumEtaMinMet_ && (jets[jetNr]).hwEta() < etSumEtaMaxMet_ && (jets[jetNr]).hwPt() > etSumEtThresholdHwMet_ )
      {
	 jetPhi=((jets[jetNr]).hwPhi()*5.0-2.5)*pi/180.;
	 coefficientX = int32_t(511.*cos(jetPhi));
	 coefficientY = int32_t(511.*sin(jetPhi));
	 htXComponent += coefficientX*ptJet;  
	 htYComponent += coefficientY*ptJet;  
      }
      if ((jets[jetNr]).hwEta() > etSumEtaMinEt_ && (jets[jetNr]).hwEta() < etSumEtaMaxEt_&& (jets[jetNr]).hwPt() > etSumEtThresholdHwEt_ )
      { 
	 totalHt += ptJet;
      } 
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

