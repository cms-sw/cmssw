///
/// \class l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetSumAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::Stage2Layer2JetSumAlgorithmFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
{
  httJetThresholdHw_ = floor(params_->etSumEtThreshold(1)/params_->jetLsb());
  mhtJetThresholdHw_ = floor(params_->etSumEtThreshold(3)/params_->jetLsb());

  httEtaMax_  = params_->etSumEtaMax(1);
  httEtaMaxHF_ = CaloTools::kHFEnd;
  mhtEtaMax_  = params_->etSumEtaMax(3);
  mhtEtaMaxHF_ = CaloTools::kHFEnd;
}


l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::~Stage2Layer2JetSumAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & alljets, std::vector<l1t::EtSum> & htsums) 
{

  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int hx(0), hy(0), ht(0);
    int hxHF(0), hyHF(0), htHF(0);

    bool satMht(false), satMhtHF(false), satHt(false), satHtHF(false);
  
    // loop over rings    
    for (unsigned absieta=1; absieta<=(unsigned int)CaloTools::mpEta(CaloTools::kHFEnd); absieta++) {

      int ieta = etaSide * absieta;

      int ringHx(0), ringHy(0), ringHt(0); 
      int ringHxHF(0), ringHyHF(0), ringHtHF(0); 
      
      // loop over phi
      for (int iphi=1; iphi<=CaloTools::kHBHENrPhi; iphi++) {
	
        // find the jet at this (eta,phi)
		l1t::Jet thisJet;
		bool foundJet = false;
		for (unsigned jetIt=0; jetIt<alljets.size(); jetIt++) {
		  if (CaloTools::mpEta(alljets.at(jetIt).hwEta())==ieta && alljets.at(jetIt).hwPhi()==iphi) {
			thisJet = alljets.at(jetIt);
			foundJet = true;
		  }
		}
		if (!foundJet) continue;
	
		  // x- and -y coefficients are truncated by after multiplication of Et by trig coefficient.
		  // The trig coefficients themselves take values [-1023,1023] and so were scaled by
		  // 2^10 = 1024, which requires bitwise shift to the right of the final value by 10 bits.
		  // The 4 below account for part of that and the rest is accounted for at ouput of demux
		  // (see Stage2Layer2DemuxSumsAlgoFirmwareImp1.cc)

		if (thisJet.hwPt()>mhtJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(mhtEtaMax_)) {
		  if(thisJet.hwPt()==CaloTools::kSatJet){
		    satMht=true;
		    satMhtHF=true;
		  }else{
		    ringHx += (int) (( thisJet.hwPt() * CaloTools::cos_coeff[iphi - 1] ) >> 4 );
		    ringHy += (int) (( thisJet.hwPt() * CaloTools::sin_coeff[iphi - 1] ) >> 4 );
		  }
		}
		if (thisJet.hwPt()>mhtJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(mhtEtaMaxHF_)) {
		  if(thisJet.hwPt()==CaloTools::kSatJet) satMhtHF=true;
		  else{
		    ringHxHF += (int) (( thisJet.hwPt() * CaloTools::cos_coeff[iphi - 1] ) >> 4 );
		    ringHyHF += (int) (( thisJet.hwPt() * CaloTools::sin_coeff[iphi - 1] ) >> 4 );
		  }
		}
		
		if (thisJet.hwPt()>httJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(httEtaMax_)) {
		  if(thisJet.hwPt()==CaloTools::kSatJet){
		    satHt=true;
		    satHtHF=true;
		  }
		  else ringHt += thisJet.hwPt();
		}
		if (thisJet.hwPt()>httJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(httEtaMaxHF_)) {
		  if(thisJet.hwPt()==CaloTools::kSatJet) satHtHF=true;
		  else ringHtHF += thisJet.hwPt();
		}
      }

      hx += ringHx;
      hy += ringHy;
      ht += ringHt;
      
      hxHF += ringHxHF;
      hyHF += ringHyHF;
      htHF += ringHtHF;

    }

    if(satHt) ht = 0xffff;
    if(satHtHF) htHF = 0xffff;

    if(satMht){ 
      hx = 0x7fffffff;
      hy = 0x7fffffff;
    }
    if(satMhtHF){
      hxHF = 0x7fffffff;
      hyHF = 0x7fffffff;
    }
    
    math::XYZTLorentzVector p4;
    
    l1t::EtSum htSumHt(p4,l1t::EtSum::EtSumType::kTotalHt,ht,0,0,0);
    l1t::EtSum htSumHx(p4,l1t::EtSum::EtSumType::kTotalHtx,hx,0,0,0);
    l1t::EtSum htSumHy(p4,l1t::EtSum::EtSumType::kTotalHty,hy,0,0,0);

    l1t::EtSum htSumHtHF(p4,l1t::EtSum::EtSumType::kTotalHtHF,htHF,0,0,0);
    l1t::EtSum htSumHxHF(p4,l1t::EtSum::EtSumType::kTotalHtxHF,hxHF,0,0,0);
    l1t::EtSum htSumHyHF(p4,l1t::EtSum::EtSumType::kTotalHtyHF,hyHF,0,0,0);
    
    htsums.push_back(htSumHt);
    htsums.push_back(htSumHx);
    htsums.push_back(htSumHy);
     
    htsums.push_back(htSumHtHF);
    htsums.push_back(htSumHxHF);
    htsums.push_back(htSumHyHF);

  }
}

