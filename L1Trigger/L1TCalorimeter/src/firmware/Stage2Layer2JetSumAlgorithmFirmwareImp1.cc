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

  httEtaMax_ = params_->etSumEtaMax(1);
  mhtEtaMax_ = params_->etSumEtaMax(3);

}


l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::~Stage2Layer2JetSumAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & alljets, std::vector<l1t::EtSum> & htsums) 
{

  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int32_t hx(0), hy(0), ht(0);
    
    // loop over rings    
    for (unsigned absieta=1; absieta<CaloTools::kHFEnd; absieta++) {

      int ieta = etaSide * absieta;

      int32_t ringHx(0), ringHy(0), ringHt(0); 

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
	
		if (thisJet.hwPt()>mhtJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=mhtEtaMax_) {

		  // x- and -y coefficients are truncated by after multiplication of Et by trig coefficient.
		  // The trig coefficients themselves take values [-1023,1023] and so were scaled by
		  // 2^10 = 1024, which requires bitwise shift to the right of the final value by 10 bits.
		  // The 4 below account for part of that and the rest is accounted for at ouput of demux
		  // (see Stage2Layer2DemuxSumsAlgoFirmwareImp1.cc)
		  ringHx += (int32_t) (( thisJet.hwPt() * CaloTools::cos_coeff[iphi - 1] ) >> 4 );
		  ringHy += (int32_t) (( thisJet.hwPt() * CaloTools::sin_coeff[iphi - 1] ) >> 4 );
		  	  
		}
	
		if (thisJet.hwPt()>httJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=httEtaMax_) {
		  ringHt += thisJet.hwPt();
		}
      }

      hx += ringHx;
      hy += ringHy;
      ht += ringHt;
      
    }

    math::XYZTLorentzVector p4;
    
    l1t::EtSum htSumHt(p4,l1t::EtSum::EtSumType::kTotalHt,ht,0,0,0);
    l1t::EtSum htSumHx(p4,l1t::EtSum::EtSumType::kTotalHtx,hx,0,0,0);
    l1t::EtSum htSumHy(p4,l1t::EtSum::EtSumType::kTotalHty,hy,0,0,0);
    
    htsums.push_back(htSumHt);
    htsums.push_back(htSumHx);
    htsums.push_back(htSumHy);
    
  }
}

