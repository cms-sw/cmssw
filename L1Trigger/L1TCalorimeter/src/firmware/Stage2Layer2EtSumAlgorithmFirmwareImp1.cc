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
  metTowThresholdHw2_ = metTowThresholdHw_;
  ettTowThresholdHw_ = floor(params_->etSumEtThreshold(2)/params_->towerLsbSum());

  metEtaMax_ = params_->etSumEtaMax(0);
  metEtaMax2_ = CaloTools::kHFEnd;
  ettEtaMax_ = params_->etSumEtaMax(2);
}


l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::~Stage2Layer2EtSumAlgorithmFirmwareImp1() {}

void l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
                                                               std::vector<l1t::EtSum> & etsums) {

  
  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int32_t ex(0), ey(0), et(0);
    int32_t ex2(0), ey2(0);
    uint32_t mb0(0), mb1(0);

    for (unsigned absieta=1; absieta<CaloTools::kHFEnd; absieta++) {

      int ieta = etaSide * absieta;

      // TODO add the eta and Et thresholds

      int32_t ringEx(0), ringEy(0), ringEt(0);
      int32_t ringEx2(0), ringEy2(0);
      uint32_t ringMB0(0), ringMB1(0);

      for (int iphi=1; iphi<=CaloTools::kHBHENrPhi; iphi++) {
      
        l1t::CaloTower tower = l1t::CaloTools::getTower(towers, ieta, iphi);
	
	


	  if (tower.hwPt()>metTowThresholdHw_ && CaloTools::mpEta(abs(tower.hwEta()))<=metEtaMax_) {
	    
	    // x- and -y coefficients are truncated by after multiplication of Et by trig coefficient.
	    // The trig coefficients themselves take values [-1023,1023] and so were scaled by
	    // 2^10 = 1024, which requires bitwise shift to the right of the final value by 10 bits.
	    // This is accounted for at ouput of demux (see Stage2Layer2DemuxSumsAlgoFirmwareImp1.cc)
	    ringEx += (int32_t) (tower.hwPt() * CaloTools::cos_coeff[iphi - 1] );
	    ringEy += (int32_t) (tower.hwPt() * CaloTools::sin_coeff[iphi - 1] );	    
	  }


	// MET no HF
	if (tower.hwPt()>metTowThresholdHw2_ && CaloTools::mpEta(abs(tower.hwEta()))<=metEtaMax2_) {
	  ringEx2 += (int32_t) (tower.hwPt() * CaloTools::cos_coeff[iphi - 1] );
	  ringEy2 += (int32_t) (tower.hwPt() * CaloTools::sin_coeff[iphi - 1] );	    
	}
	
	// scalar sum
	if (tower.hwPt()>ettTowThresholdHw_ && CaloTools::mpEta(abs(tower.hwEta()))<=ettEtaMax_) 
	  ringEt += tower.hwPt();
		
	// count HF tower HCAL flags
	if (CaloTools::mpEta(abs(tower.hwEta()))>CaloTools::kHFBegin &&
	    CaloTools::mpEta(abs(tower.hwEta()))<CaloTools::kHFEnd &&
	    (tower.hwQual() & 0x4) > 0) 
	  ringMB1 += 1;
	
      }    
      
      ex += ringEx;
      ey += ringEy;
      et += ringEt;
      ex2 += ringEx2;
      ey2 += ringEy2;
      mb0 += ringMB0;
      mb1 += ringMB1;

    }

    if (mb0>0xf) mb0 = 0xf;
    if (mb1>0xf) mb1 = 0xf;

    math::XYZTLorentzVector p4;

    l1t::EtSum etSumTotalEt(p4,l1t::EtSum::EtSumType::kTotalEt,et,0,0,0);
    l1t::EtSum etSumEx(p4,l1t::EtSum::EtSumType::kTotalEtx,ex,0,0,0);
    l1t::EtSum etSumEy(p4,l1t::EtSum::EtSumType::kTotalEty,ey,0,0,0);
    l1t::EtSum etSumEx2(p4,l1t::EtSum::EtSumType::kTotalEtx2,ex2,0,0,0);
    l1t::EtSum etSumEy2(p4,l1t::EtSum::EtSumType::kTotalEty2,ey2,0,0,0);

    l1t::EtSum::EtSumType type0 = l1t::EtSum::EtSumType::kMinBiasHFP0;
    l1t::EtSum::EtSumType type1 = l1t::EtSum::EtSumType::kMinBiasHFP1;
    if (etaSide<0) {
      type0 = l1t::EtSum::EtSumType::kMinBiasHFM0;
      type1 = l1t::EtSum::EtSumType::kMinBiasHFM1;
    } 
    l1t::EtSum etSumMinBias0(p4,type0,mb0,0,0,0);
    l1t::EtSum etSumMinBias1(p4,type1,mb1,0,0,0);

    etsums.push_back(etSumTotalEt);
    etsums.push_back(etSumEx);
    etsums.push_back(etSumEy);
    etsums.push_back(etSumEx2);
    etsums.push_back(etSumEy2);
    etsums.push_back(etSumMinBias0);
    etsums.push_back(etSumMinBias1);

  }

}
