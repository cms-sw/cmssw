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
      
      int iet(0);
      
      // loop over towers in this jet
      for( int deta = 0; deta != 8; ++deta ) {
	for( int dphi = 0; dphi != 8; ++dphi ) {
	  
	  // check jet mask and sum tower et
	  // re-use calo tools sum method, but for single tower
	  if( mask[deta][dphi] )
	    iet += CaloTools::calHwEtSum(ieta, iphi, towers,
					   0, 0, 0, 0, CaloTools::CALO);
	  
	}
      }

      // add the jet to the list
      math::XYZTLorentzVector p4;
      l1t::Jet jet( p4, iet, ieta, iphi, 0);
      jets.push_back( jet );
      
    }
  }
  
    
}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::filter(std::vector<l1t::Jet> & jets) {

  // do nothing for now!

}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::sort(std::vector<l1t::Jet> & jets) {

  // do nothing for now!

}

