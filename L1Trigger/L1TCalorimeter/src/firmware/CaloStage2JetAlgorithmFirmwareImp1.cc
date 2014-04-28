///
/// \class l1t::CaloStage2JetAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetAlgorithmFirmware.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include <vector>

l1t::CaloStage2JetAlgorithmFirmwareImp1::CaloStage2JetAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::CaloStage2JetAlgorithmFirmwareImp1::~CaloStage2JetAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
							      std::vector<l1t::Jet> & jets) {


  // find all possible jets
  create(towers, jets);

  // remove overlaps
  filter(jets);

  // sort
  sort(jets);

}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::create(const std::vector<l1t::CaloTower> & towers,
						     std::vector<l1t::Jet> & jets) {

  
  // generate jet mask
  // needs to be configurable at some point
  bool mask[8][8] = {
    { 0,0,1,1,1,1,0,0 },
    { 0,1,1,1,1,1,1,0 },
    { 1,1,1,1,1,1,1,1 },
    { 1,1,1,1,1,1,1,1 },
    { 1,1,1,1,1,1,1,1 },
    { 1,1,1,1,1,1,1,1 },
    { 0,1,1,1,1,1,1,0 },
    { 0,0,1,1,1,1,0,0 }
  };
  

  // loop over jet positions
  for ( int ieta = -27 ; ieta != 27 ; ++ieta ) {
    if (ieta==0) continue;
    for ( int iphi = 0 ; iphi != 72 ; ++iphi ) {
      
      int iEt(0);
      const CaloTower& tow = CaloTools::getTower(towers, ieta, iphi); 
      int seedEt = tow.hwEtEm();
      seedEt += tow.hwEtHad();
      bool isMax(true);

      // loop over towers in this jet
      for( int deta = -4; deta != 4; ++deta ) {
	for( int dphi = -4; dphi != 4; ++dphi ) {
	  
	  // check jet mask and sum tower et
	  // re-use calo tools sum method, but for single tower
	  if( mask[deta+4][dphi+4] ) {
	    const CaloTower& tow = CaloTools::getTower(towers, ieta+deta, iphi+dphi); 
	    int towEt = tow.hwEtEm() + tow.hwEtHad();
	    iEt+=towEt;
	    isMax=(seedEt>towEt);
	  }
	}
      }

      // add the jet to the list
      if (iEt>params_->jetSeedThreshold() && isMax) {
	math::XYZTLorentzVector p4;
	l1t::Jet jet( p4, iEt, ieta, iphi, 0);
	jets.push_back( jet );
      }

    }
  }
  
}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::filter(std::vector<l1t::Jet> & jets) {

  //  jets.erase(std::remove_if(jets.begin(), jets.end(), jetIsZero) );

}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::sort(std::vector<l1t::Jet> & jets) {

  // do nothing for now!

}

  // remove jets with zero et
bool l1t::CaloStage2JetAlgorithmFirmwareImp1::jetIsZero(l1t::Jet jet) { return jet.hwPt()==0; }

