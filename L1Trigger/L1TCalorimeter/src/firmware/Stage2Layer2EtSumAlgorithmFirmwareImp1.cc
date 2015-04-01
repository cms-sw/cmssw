///
/// \class l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EtSumAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include <math.h>


l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::Stage2Layer2EtSumAlgorithmFirmwareImp1(CaloParams* params) :
   params_(params)
{

  // Add some LogDebug for these settings

  etSumEtThresholdHwEt_ = floor(params_->etSumEtThreshold(0)/params_->towerLsbSum());
  etSumEtThresholdHwMet_ = floor(params_->etSumEtThreshold(2)/params_->towerLsbSum());
  
  etSumEtaMinEt_ = params_->etSumEtaMin(0);
  etSumEtaMaxEt_ = params_->etSumEtaMax(0);
  
  etSumEtaMinMet_ = params_->etSumEtaMin(2);
  etSumEtaMaxMet_ = params_->etSumEtaMax(2);
}


l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::~Stage2Layer2EtSumAlgorithmFirmwareImp1() {}

void l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
                                                               std::vector<l1t::EtSum> & etsums) {
  
  int etaMax=40, etaMin=1, phiMax=72, phiMin=1;
  
  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int32_t ex(0), ey(0), et(0);
    
    std::vector<int> rings;
    for (int i=etaMin; i<=etaMax; i++) rings.push_back(i*etaSide);
    
    for (unsigned etaIt=0; etaIt<rings.size(); etaIt++) {
      
      int ieta = rings.at(etaIt);
         
      // TODO add the eta and Et thresholds
      
      int32_t ringEx(0), ringEy(0), ringEt(0);
      
      for (int iphi=phiMin; iphi<=phiMax; iphi++) {
	
	l1t::CaloTower tower = l1t::CaloTools::getTower(towers, ieta, iphi);
	
	ringEx += (int32_t) (tower.hwPt() * std::trunc ( 511. * cos ( 2 * M_PI * (72 - iphi) / 72.0 ) )) >> 9;
	ringEy += (int32_t) (tower.hwPt() * std::trunc ( 511. * sin ( 2 * M_PI * iphi / 72.0 ) )) >> 9;
	ringEt += tower.hwPt();
	
      }
      
      // At some point we will remove the bit shifts and will need to limit to the precision available in the firmware
      
      ex += ( ringEx >> 2);
      ey += ( ringEy >> 2);
      et += ( ringEt >> 1);
      
    }
    
    math::XYZTLorentzVector p4;

    l1t::EtSum etSumTotalEt(p4,l1t::EtSum::EtSumType::kTotalEt,et,0,0,0);
    l1t::EtSum etSumEx(p4,l1t::EtSum::EtSumType::kTotalEtx,ex,0,0,0);
    l1t::EtSum etSumEy(p4,l1t::EtSum::EtSumType::kTotalEty,ey,0,0,0);
    
    etsums.push_back(etSumTotalEt);
    etsums.push_back(etSumEx);
    etsums.push_back(etSumEy);

  }
  
}
