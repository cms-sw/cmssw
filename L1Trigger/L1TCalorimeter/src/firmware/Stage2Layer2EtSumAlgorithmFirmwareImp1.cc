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


l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::Stage2Layer2EtSumAlgorithmFirmwareImp1(CaloParamsHelper* params) :
   params_(params)
{

  // Add some LogDebug for these settings

  metTowThresholdHw_ = floor(params_->etSumEtThreshold(0)/params_->towerLsbSum());
  ettTowThresholdHw_ = floor(params_->etSumEtThreshold(2)/params_->towerLsbSum());

  metEtaMax_ = params_->etSumEtaMax(0);
  ettEtaMax_ = params_->etSumEtaMax(2);
}


l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::~Stage2Layer2EtSumAlgorithmFirmwareImp1() {}

void l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
                                                               std::vector<l1t::EtSum> & etsums) {

  
  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int32_t ex(0), ey(0), et(0);

    for (unsigned absieta=1; absieta<CaloTools::kHFEnd; absieta++) {

      int ieta = etaSide * absieta;

      // TODO add the eta and Et thresholds

      int32_t ringEx(0), ringEy(0), ringEt(0);

      for (int iphi=1; iphi<=CaloTools::kHBHENrPhi; iphi++) {
      
        l1t::CaloTower tower = l1t::CaloTools::getTower(towers, ieta, iphi);

	if (tower.hwPt()>metTowThresholdHw_ && CaloTools::mpEta(abs(tower.hwEta()))<=metEtaMax_) {
	  ringEx += (int32_t) (tower.hwPt() * std::trunc ( 1023. * cos ( 2 * M_PI * (72 - (iphi-1)) / 72.0 ) ));
	  ringEy += (int32_t) (tower.hwPt() * std::trunc ( 1023. * sin ( 2 * M_PI * (iphi-1) / 72.0 ) ));

	}
	if (tower.hwPt()>ettTowThresholdHw_ && CaloTools::mpEta(abs(tower.hwEta()))<=ettEtaMax_) 
	  ringEt += tower.hwPt();
      }    
      
      ex += ringEx;
      ey += ringEy;
      et += ringEt;
    }

    ex >>= 10;
    ey >>= 10;

    math::XYZTLorentzVector p4;

    l1t::EtSum etSumTotalEt(p4,l1t::EtSum::EtSumType::kTotalEt,et,0,0,0);
    l1t::EtSum etSumEx(p4,l1t::EtSum::EtSumType::kTotalEtx,ex,0,0,0);
    l1t::EtSum etSumEy(p4,l1t::EtSum::EtSumType::kTotalEty,ey,0,0,0);

    etsums.push_back(etSumTotalEt);
    etsums.push_back(etSumEx);
    etsums.push_back(etSumEy);

  }

}
