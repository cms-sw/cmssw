
///
/// \class l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EtSumAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include <cmath>


l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::Stage2Layer2EtSumAlgorithmFirmwareImp1(CaloParamsHelper const* params) :
  params_(params)
{

  // Add some LogDebug for these settings
  metEtaMax_ = params_->etSumEtaMax(2);
  metEtaMaxHF_ = CaloTools::kHFEnd;
  ettEtaMax_ = params_->etSumEtaMax(0);
  ettEtaMaxHF_ = CaloTools::kHFEnd;

  nTowThresholdHw_ = floor(params_->etSumEtThreshold(4)/params_->towerLsbSum());
  nTowEtaMax_ = params_->etSumEtaMax(4);
}


void l1t::Stage2Layer2EtSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
                                                               std::vector<l1t::EtSum> & etsums) {

  unsigned int ntowers(0);
  math::XYZTLorentzVector p4;
  
  int nTT4 = CaloTools::calNrTowers(-1*params_->egPUSParam(1),
				    params_->egPUSParam(1),
				    1,72,towers,1+params_->pileUpTowerThreshold(),999,CaloTools::CALO);
  unsigned int compNTT4 = params_->egCompressShapesLUT()->data((0x1<<7)+(0x1<<8)+(0x1<<5)+nTT4);


  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int ex(0), ey(0), et(0);
    int exHF(0), eyHF(0), etHF(0);
    int etem(0);
    unsigned int mb0(0), mb1(0);

    bool ettSat(false), ettHFSat(false), ecalEtSat(false), metSat(false), metHFSat(false);

    for (unsigned absieta=1; absieta<=(unsigned int)CaloTools::mpEta(CaloTools::kHFEnd); absieta++) {

      int ieta = etaSide * absieta;
      
      towEtMetThresh_ = 0;
      towEtSumEtThresh_ = 0;
      towEtEcalSumThresh_ = 0;

      if(!params_->etSumBypassMetPUS()){
	if(params_->etSumMetPUSType() == "LUT"){
	  unsigned int towEtMetLUTAddr = (compNTT4<<6) | (abs(ieta));
	  if(abs(ieta)<13) towEtMetLUTAddr = abs(ieta);
	  towEtMetThresh_ = params_->etSumMetPUSLUT()->data(towEtMetLUTAddr);
	} else {
	  if(params_->etSumMetPUSType() != "None" && params_->etSumMetPUSType() != "none") {
	    edm::LogError("l1t|stage 2") << "Invalid PUS type in calo params. Not applying PUS to Stage 2 MET" << std::endl;
	      return;
	  }
	}
      }
      
      if(!params_->etSumBypassEttPUS()){
	if(params_->etSumEttPUSType() == "LUT"){
	  unsigned int towEtSumEtLUTAddr = (compNTT4<<6) | (abs(ieta));
	  if(abs(ieta)<13) towEtSumEtLUTAddr = abs(ieta);
	  towEtSumEtThresh_ = params_->etSumEttPUSLUT()->data(towEtSumEtLUTAddr);
	} else {
	  if(params_->etSumEttPUSType() != "None" && params_->etSumEttPUSType() != "none") {
	      edm::LogError("l1t|stage 2") << "Invalid PUS type in calo params. Not applying PUS to Stage 2 ETT" << std::endl;
	      return;
	    }
	}
      }
      
      if(!params_->etSumBypassEcalSumPUS()){
	if(params_->etSumEcalSumPUSType() == "LUT"){
	  unsigned int towEtEcalSumLUTAddr = (compNTT4<<6) | (abs(ieta));
	  if(abs(ieta)<13) towEtEcalSumLUTAddr = abs(ieta);
	  towEtEcalSumThresh_ = params_->etSumEcalSumPUSLUT()->data(towEtEcalSumLUTAddr);
	} else {
	  if(params_->etSumEcalSumPUSType() != "None" && params_->etSumEcalSumPUSType() != "none") {
	    edm::LogError("l1t|stage 2") << "Invalid PUS type in calo params. Not applying PUS to Stage 2 Ecal ETT" << std::endl;
	    return;
	  }
	}
      }
      
      int ringEx(0), ringEy(0), ringEt(0);
      int ringExHF(0), ringEyHF(0), ringEtHF(0);
      int ringEtEm(0);
      unsigned int ringMB0(0), ringMB1(0);
      unsigned int ringNtowers(0);

      for (int iphi=1; iphi<=CaloTools::kHBHENrPhi; iphi++) {

        l1t::CaloTower tower = l1t::CaloTools::getTower(towers, CaloTools::caloEta(ieta), iphi);


	// MET without HF

	if (tower.hwPt()>towEtMetThresh_ && CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(metEtaMax_) && !metSat) {

	  // x- and -y coefficients are truncated by after multiplication of Et by trig coefficient.
	  // The trig coefficients themselves take values [-1023,1023] and so were scaled by
	  // 2^10 = 1024, which requires bitwise shift to the right of the final value by 10 bits.
	  // This is accounted for at ouput of demux (see Stage2Layer2DemuxSumsAlgoFirmwareImp1.cc)
	  if(tower.hwPt() == CaloTools::kSatHcal || tower.hwPt() == CaloTools::kSatEcal || tower.hwPt() == CaloTools::kSatTower) metSat=true;
	  ringEx += (int) (tower.hwPt() * CaloTools::cos_coeff[iphi - 1] );
	  ringEy += (int) (tower.hwPt() * CaloTools::sin_coeff[iphi - 1] );	    
	}

	// MET *with* HF
	if (tower.hwPt()>towEtMetThresh_ && CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(metEtaMaxHF_) && !metHFSat) {
	  if(tower.hwPt() == CaloTools::kSatHcal || tower.hwPt() == CaloTools::kSatEcal || tower.hwPt() == CaloTools::kSatTower) metHFSat=true;
	  ringExHF += (int) (tower.hwPt() * CaloTools::cos_coeff[iphi - 1] );
	  ringEyHF += (int) (tower.hwPt() * CaloTools::sin_coeff[iphi - 1] );	    
	}

	// scalar sum
	if (tower.hwPt()>towEtSumEtThresh_ && CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(ettEtaMax_) && !ettSat){
	  if(tower.hwPt() == CaloTools::kSatHcal || tower.hwPt() == CaloTools::kSatEcal || tower.hwPt() == CaloTools::kSatTower) ettSat=true;
	  ringEt += tower.hwPt();
	}
  
	// scalar sum including HF
	if (tower.hwPt()>towEtSumEtThresh_ && CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(ettEtaMaxHF_) && !ettHFSat) {
	  if(tower.hwPt() == CaloTools::kSatHcal || tower.hwPt() == CaloTools::kSatEcal || tower.hwPt() == CaloTools::kSatTower) ettHFSat=true;
	  ringEtHF += tower.hwPt();
	}
	
        // scalar sum (EM)
        if (tower.hwPt()>towEtEcalSumThresh_ && CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(ettEtaMax_) && !ecalEtSat){
	  if(tower.hwPt() == CaloTools::kSatEcal || tower.hwPt() == CaloTools::kSatTower) ecalEtSat=true;
          ringEtEm += tower.hwEtEm();
	}

	// count HF tower HCAL flags
	if (CaloTools::mpEta(abs(tower.hwEta()))>=CaloTools::mpEta(CaloTools::kHFBegin) &&
	    CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(CaloTools::kHFEnd) &&
	    (tower.hwQual() & 0x4) > 0) 
	  ringMB0 += 1;
	if (CaloTools::mpEta(abs(tower.hwEta()))>=CaloTools::mpEta(CaloTools::kHFBegin) &&
	    CaloTools::mpEta(abs(tower.hwEta()))<=CaloTools::mpEta(CaloTools::kHFEnd) &&
	    (tower.hwQual() & 0x8) > 0) 
	  ringMB1 += 1;
	
        // tower counting 
	if (tower.hwPt()>nTowThresholdHw_ && CaloTools::mpEta(abs(tower.hwEta()))<=nTowEtaMax_) 
	  ringNtowers += 1;
      }    
      
      ex += ringEx;
      ey += ringEy;
      et += ringEt;
      etHF += ringEtHF;
      exHF += ringExHF;
      eyHF += ringEyHF;

      etem  += ringEtEm;

      mb0 += ringMB0;
      mb1 += ringMB1;

      ntowers += ringNtowers;
    }

    if (mb0>0xf) mb0 = 0xf;
    if (mb1>0xf) mb1 = 0xf;


    // saturate energy sums if saturated TP/tower

    if(ecalEtSat || etem > 0xffff) etem = 0xffff;
    if(ettSat || et > 0xffff) et = 0xffff;
    if(ettHFSat || etHF > 0xffff) etHF = 0xffff;
    if(metSat || ex > 0x7fffffff) ex = 0x7fffffff;
    if(metSat || ey > 0x7fffffff) ey = 0x7fffffff;
    if(metHFSat || exHF > 0x7fffffff) exHF = 0x7fffffff;
    if(metHFSat || eyHF > 0x7fffffff) eyHF = 0x7fffffff;
    
    l1t::EtSum etSumTotalEt(p4,l1t::EtSum::EtSumType::kTotalEt,et,0,0,0);
    l1t::EtSum etSumEx(p4,l1t::EtSum::EtSumType::kTotalEtx,ex,0,0,0);
    l1t::EtSum etSumEy(p4,l1t::EtSum::EtSumType::kTotalEty,ey,0,0,0);

    l1t::EtSum etSumTotalEtHF(p4,l1t::EtSum::EtSumType::kTotalEtHF,etHF,0,0,0);
    l1t::EtSum etSumExHF(p4,l1t::EtSum::EtSumType::kTotalEtxHF,exHF,0,0,0);
    l1t::EtSum etSumEyHF(p4,l1t::EtSum::EtSumType::kTotalEtyHF,eyHF,0,0,0);

    l1t::EtSum etSumTotalEtEm(p4,l1t::EtSum::EtSumType::kTotalEtEm,etem,0,0,0);

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

    etsums.push_back(etSumTotalEtHF);
    etsums.push_back(etSumExHF);
    etsums.push_back(etSumEyHF);

    etsums.push_back(etSumTotalEtEm);

    etsums.push_back(etSumMinBias0);
    etsums.push_back(etSumMinBias1);

  }

  //tower count is in aux: only on eta- side!!
  l1t::EtSum etSumNtowers(p4,l1t::EtSum::EtSumType::kTowerCount,ntowers,0,0,0);
  etsums.push_back(etSumNtowers);

}
