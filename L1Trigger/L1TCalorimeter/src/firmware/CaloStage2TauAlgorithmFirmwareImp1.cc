///
/// \class l1t::CaloStage2TauAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TauAlgorithmFirmware.h"



l1t::CaloStage2TauAlgorithmFirmwareImp1::CaloStage2TauAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::CaloStage2TauAlgorithmFirmwareImp1::~CaloStage2TauAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
							      std::vector<l1t::Tau> & taus) {


  for ( auto itr = clusters.cbegin(); itr != clusters.end(); ++itr ) {
    math::XYZTLorentzVector p4;
    l1t::Tau tau( p4, itr->hwPt(), itr->hwEta(), itr->hwPhi(), 0);
    taus.push_back(tau);
  }

}

