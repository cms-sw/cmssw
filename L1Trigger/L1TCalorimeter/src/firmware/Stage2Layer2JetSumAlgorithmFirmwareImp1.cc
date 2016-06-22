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
  httEtaMax2_ = CaloTools::kHFEnd;
  mhtEtaMax_  = params_->etSumEtaMax(3);
  mhtEtaMax2_ = CaloTools::kHFEnd;
}


l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::~Stage2Layer2JetSumAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & alljets, std::vector<l1t::EtSum> & htsums) 
{

  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int32_t hx(0), hy(0), ht(0);
    int32_t hx2(0), hy2(0), ht2(0);
  
    // loop over rings    
    for (unsigned absieta=1; absieta<=(uint)CaloTools::mpEta(CaloTools::kHFEnd); absieta++) {

      int ieta = etaSide * absieta;

      int32_t ringHx(0), ringHy(0), ringHt(0); 
      int32_t ringHx2(0), ringHy2(0), ringHt2(0); 
      
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
		  ringHx += ((CaloTools::cos_coeff[iphi - 1] > 0 ) - (CaloTools::cos_coeff[iphi - 1] < 0 )) * (int32_t) (( (uint64_t)(thisJet.hwPt() * abs(CaloTools::cos_coeff[iphi - 1])) ) >> 4 );
		  ringHy += ((CaloTools::sin_coeff[iphi - 1] > 0 ) - (CaloTools::sin_coeff[iphi - 1] < 0 )) * (int32_t) (( (uint64_t)(thisJet.hwPt() * abs(CaloTools::sin_coeff[iphi - 1])) ) >> 4 );
		}
		if (thisJet.hwPt()>mhtJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(mhtEtaMax2_)) {
		  ringHx2 += ((CaloTools::cos_coeff[iphi - 1] > 0 ) - (CaloTools::cos_coeff[iphi - 1] < 0 )) * (int32_t) (( (uint64_t)(thisJet.hwPt() * abs(CaloTools::cos_coeff[iphi - 1])) ) >> 4 );
		  ringHy2 += ((CaloTools::sin_coeff[iphi - 1] > 0 ) - (CaloTools::sin_coeff[iphi - 1] < 0 )) * (int32_t) (( (uint64_t)(thisJet.hwPt() * abs(CaloTools::sin_coeff[iphi - 1])) ) >> 4 );
		}
		
		if (thisJet.hwPt()>httJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(httEtaMax_)) {
		  ringHt += thisJet.hwPt();
		}
		if (thisJet.hwPt()>httJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=CaloTools::mpEta(httEtaMax2_)) {
		  ringHt2 += thisJet.hwPt();
		}
      }

      hx += ringHx;
      hy += ringHy;
      ht += ringHt;
      
      hx2 += ringHx2;
      hy2 += ringHy2;
      ht2 += ringHt2;

    }

    math::XYZTLorentzVector p4;
    
    l1t::EtSum htSumHt(p4,l1t::EtSum::EtSumType::kTotalHt,ht,0,0,0);
    l1t::EtSum htSumHx(p4,l1t::EtSum::EtSumType::kTotalHtx,hx,0,0,0);
    l1t::EtSum htSumHy(p4,l1t::EtSum::EtSumType::kTotalHty,hy,0,0,0);

    l1t::EtSum htSumHt2(p4,l1t::EtSum::EtSumType::kTotalHt2,ht2,0,0,0);
    l1t::EtSum htSumHx2(p4,l1t::EtSum::EtSumType::kTotalHtx2,hx2,0,0,0);
    l1t::EtSum htSumHy2(p4,l1t::EtSum::EtSumType::kTotalHty2,hy2,0,0,0);
    
    htsums.push_back(htSumHt);
    htsums.push_back(htSumHx);
    htsums.push_back(htSumHy);
     
    htsums.push_back(htSumHt2);
    htsums.push_back(htSumHx2);
    htsums.push_back(htSumHy2);
    
  }
}

