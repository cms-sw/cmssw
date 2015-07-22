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
  etSumEtThresholdHwEt_ = floor(params_->etSumEtThreshold(0)/params_->jetLsb());
  etSumEtThresholdHwMet_ = floor(params_->etSumEtThreshold(2)/params_->jetLsb());

  etSumEtaMinEt_ = params_->etSumEtaMin(0);
  etSumEtaMaxEt_ = params_->etSumEtaMax(0);
 
  etSumEtaMinMet_ = params_->etSumEtaMin(2);
  etSumEtaMaxMet_ = params_->etSumEtaMax(2);
}


l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::~Stage2Layer2JetSumAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetSumAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::Jet> & alljets, std::vector<l1t::EtSum> & htsums) 
{

  int etaMax=40, etaMin=1, phiMax=72, phiMin=1;
  
  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {

    int32_t hx(0), hy(0), ht(0);
    
    std::vector<int> rings;
    for (int i=etaMin; i<=etaMax; i++) rings.push_back(i*etaSide);

    // loop over rings    
    for (unsigned etaIt=0; etaIt<rings.size(); etaIt++) {

      int ieta = rings.at(etaIt);

      int32_t ringHx(0), ringHy(0), ringHt(0); 

      // loop over phi
      for (int iphi=phiMin; iphi<=phiMax; iphi++) {
	
        // find the jet at this (eta,phi)
	l1t::Jet thisJet;
	bool foundJet = false;
	for (unsigned jetIt=0; jetIt<alljets.size(); jetIt++) {
	  if (alljets.at(jetIt).hwEta()==ieta && alljets.at(jetIt).hwPhi()==iphi) {
	    thisJet = alljets.at(jetIt);
	    foundJet = true;
	  }
	}
	if (!foundJet) continue;
	
	if (thisJet.hwPt()>etSumEtThresholdHwMet_ && thisJet.hwEta()>=etSumEtaMinMet_ && thisJet.hwEta()<=etSumEtaMaxMet_) {
	  ringHx += (int32_t) ( thisJet.hwPt() * std::trunc ( 511. * cos ( 2 * M_PI * (72 - iphi) / 72.0 ) )) >> 10;
	  ringHy += (int32_t) ( thisJet.hwPt() * std::trunc ( 511. * sin ( 2 * M_PI * iphi / 72.0 ) )) >> 10;
	}
	
	if (thisJet.hwPt()>etSumEtThresholdHwEt_ && thisJet.hwEta()>=etSumEtaMinEt_ && thisJet.hwEta()<=etSumEtaMaxEt_) {
	  ringHt += thisJet.hwPt();
	}
      }

      hx += (ringHx >> 5);
      hy += (ringHy >> 5);
      ht += (ringHt >> 5);
      
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

