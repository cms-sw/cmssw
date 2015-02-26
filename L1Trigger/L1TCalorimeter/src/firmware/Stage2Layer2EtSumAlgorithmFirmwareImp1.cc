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

  int ietaMax=40, ietaMin=-40, iphiMax=72, iphiMin=1;

  int32_t ex_poseta(0), ey_poseta(0), et_poseta(0);
  int32_t ex_negeta(0), ey_negeta(0), et_negeta(0);

  for (int ieta=ietaMin; ieta<=ietaMax; ieta++) {

    // Add the eta and Et thresholds

    if (ieta==0) continue;

    int32_t ringEx(0), ringEy(0), ringEt(0);

    for (int iphi=iphiMin; iphi<=iphiMax; iphi++) {
      
      l1t::CaloTower tower = l1t::CaloTools::getTower(towers, ieta, iphi);

      ringEx += (int32_t) (tower.hwPt() * std::trunc ( 511. * cos ( 2 * M_PI * (72 - (iphi-1)) / 72.0 ) )) >> 9;
      ringEy += (int32_t) (tower.hwPt() * std::trunc ( 511. * sin ( 2 * M_PI * (iphi-1) / 72.0 ) )) >> 9;
      ringEt += tower.hwPt();
      
    }
    
    // At some point we will remove the bit shifts and will need to limit to the precision available in the firmware
    
    if (ieta>0) { // Positive eta
      ex_poseta += ( ringEx >> 2);
      ey_poseta += ( ringEy >> 2);
      et_poseta += ( ringEt >> 1);
    } else {    // Negative eta
      ex_negeta += ( ringEx >> 2);
      ey_negeta += ( ringEy >> 2);
      et_negeta += ( ringEt >> 1);
    }

    //Hack before bit shifts are decided
    //et+=ringEt;
    //ex+=ringEx;
    //ey+=ringEy;

  }

  // push output
  math::XYZTLorentzVector p4;

  l1t::EtSum etSumTotalEtPosEta(p4,l1t::EtSum::EtSumType::kTotalEt,et_poseta,0,0,0);
  l1t::EtSum etSumTotalEtNegEta(p4,l1t::EtSum::EtSumType::kTotalEt,et_negeta,0,0,0);
  l1t::EtSum etSumExPosEta(p4,l1t::EtSum::EtSumType::kTotalEtx,ex_poseta,0,0,0);
  l1t::EtSum etSumExNegEta(p4,l1t::EtSum::EtSumType::kTotalEtx,ex_negeta,0,0,0);
  l1t::EtSum etSumEyPosEta(p4,l1t::EtSum::EtSumType::kTotalEty,ey_poseta,0,0,0);
  l1t::EtSum etSumEyNegEta(p4,l1t::EtSum::EtSumType::kTotalEty,ey_negeta,0,0,0);

  etsums.push_back(etSumTotalEtPosEta);
  etsums.push_back(etSumTotalEtNegEta);
  etsums.push_back(etSumExPosEta);
  etsums.push_back(etSumExNegEta);
  etsums.push_back(etSumEyPosEta);
  etsums.push_back(etSumEyNegEta);

}

