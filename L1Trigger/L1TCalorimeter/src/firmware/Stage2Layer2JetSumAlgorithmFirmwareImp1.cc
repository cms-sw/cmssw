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
	  ringHx += (int32_t) ( thisJet.hwPt() * std::trunc ( 1023. * cos ( 2 * M_PI * (72 - (iphi-1)) / 72.0 ) ));
	  ringHy += (int32_t) ( thisJet.hwPt() * std::trunc ( 1023. * sin ( 2 * M_PI * (iphi-1) / 72.0 ) ));
	}
	
	if (thisJet.hwPt()>httJetThresholdHw_ && CaloTools::mpEta(abs(thisJet.hwEta()))<=httEtaMax_) {
	  ringHt += thisJet.hwPt();
	}
      }

      hx += ringHx;
      hy += ringHy;
      ht += ringHt;
      
    }

    hx >>= 10;
    hy >>= 10;

    math::XYZTLorentzVector p4;
    
    l1t::EtSum htSumHt(p4,l1t::EtSum::EtSumType::kTotalHt,ht,0,0,0);
    l1t::EtSum htSumHx(p4,l1t::EtSum::EtSumType::kTotalHtx,hx,0,0,0);
    l1t::EtSum htSumHy(p4,l1t::EtSum::EtSumType::kTotalHty,hy,0,0,0);
    
    htsums.push_back(htSumHt);
    htsums.push_back(htSumHx);
    htsums.push_back(htSumHy);
    
  }
}

