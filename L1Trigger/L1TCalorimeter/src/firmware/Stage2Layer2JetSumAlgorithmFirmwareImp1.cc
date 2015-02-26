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
  etSumEtThresholdHwEt_ = 30;//floor(params_->etSumEtThreshold(1)/params_->jetLsb());
  etSumEtThresholdHwMet_ = 30;//floor(params_->etSumEtThreshold(3)/params_->jetLsb());

  etSumEtaMinEt_ = params_->etSumEtaMin(1);
  etSumEtaMaxEt_ = params_->etSumEtaMax(1);
 
  etSumEtaMinMet_ = params_->etSumEtaMin(3);
  etSumEtaMaxMet_ = params_->etSumEtaMax(3);
}


l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::~Stage2Layer2JetSumAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & jets,
							      std::vector<l1t::EtSum> & etsums) 
{
   int32_t ht_poseta(0), hx_poseta(0), hy_poseta(0); 
   int32_t ht_negeta(0), hx_negeta(0), hy_negeta(0);

   for(std::vector<l1t::Jet>::const_iterator lIt = jets.begin() ; lIt != jets.end() ; ++lIt )
     {

       // Positive eta
       if (lIt->hwEta()>0) { 
	 if (lIt->hwPt()>etSumEtThresholdHwMet_ && lIt->hwEta() >= etSumEtaMinMet_ && lIt->hwEta() <= etSumEtaMaxMet_){
	   hy_poseta += (int32_t) ( lIt->hwPt() * std::trunc ( 511. * cos ( 6.28318530717958647693 * (72 - ( lIt->hwPhi() - 1 )) / 72.0 ) )) >> 9;
	   hx_poseta += (int32_t) ( lIt->hwPt() * std::trunc ( 511. * sin ( 6.28318530717958647693 * ( lIt->hwPhi() - 1 ) / 72.0 ) )) >> 9;
	 }
	 if (lIt->hwPt()>etSumEtThresholdHwEt_ && lIt->hwEta() >= etSumEtaMinEt_ && lIt->hwEta() <= etSumEtaMaxEt_){
	   ht_poseta += lIt->hwPt();
	 }
       } 
       
       // Negative eta
       else {              
	 if (lIt->hwPt()>etSumEtThresholdHwMet_ && lIt->hwEta() >= etSumEtaMinMet_ && lIt->hwEta() <= etSumEtaMaxMet_){
	   hy_negeta += (int32_t) ( lIt->hwPt() * std::trunc ( 511. * cos ( 6.28318530717958647693 * (72 - ( lIt->hwPhi() - 1 )) / 72.0 ) )) >> 9;
	   hx_negeta += (int32_t) ( lIt->hwPt() * std::trunc ( 511. * sin ( 6.28318530717958647693 * ( lIt->hwPhi() - 1 ) / 72.0 ) )) >> 9;
	 }
	 if (lIt->hwPt()>etSumEtThresholdHwEt_ && lIt->hwEta() >= etSumEtaMinEt_ && lIt->hwEta() <= etSumEtaMaxEt_){
	   ht_negeta += lIt->hwPt();
	 }
       }

     }

   hx_poseta >>=5;
   hy_poseta >>=5;
   ht_poseta >>=5;
   hx_negeta >>=5;
   hy_negeta >>=5;
   ht_negeta >>=5;
   
   math::XYZTLorentzVector p4;
   
   l1t::EtSum htSumhtPosEta( p4 , l1t::EtSum::EtSumType::kTotalHt ,ht_poseta,0,0,0);
   l1t::EtSum htSumhtNegEta( p4 , l1t::EtSum::EtSumType::kTotalHt ,ht_negeta,0,0,0);
   l1t::EtSum htSumMissingHtxPosEta( p4 , l1t::EtSum::EtSumType::kTotalHtx ,hx_poseta,0,0,0);
   l1t::EtSum htSumMissingHtxNegEta( p4 , l1t::EtSum::EtSumType::kTotalHtx ,hx_negeta,0,0,0);
   l1t::EtSum htSumMissingHtyPosEta( p4 , l1t::EtSum::EtSumType::kTotalHty ,hy_poseta,0,0,0);
   l1t::EtSum htSumMissingHtyNegEta( p4 , l1t::EtSum::EtSumType::kTotalHty ,hy_negeta,0,0,0);

   etsums.push_back(htSumhtPosEta);
   etsums.push_back(htSumhtNegEta);
   etsums.push_back(htSumMissingHtxPosEta);
   etsums.push_back(htSumMissingHtxNegEta);
   etsums.push_back(htSumMissingHtyPosEta);
   etsums.push_back(htSumMissingHtyNegEta);
}

