///
/// \class l1t::CaloStage2TowerAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TowerAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "

#include "CondFormats/L1TObjects/interface/CaloParams.h"

l1t::CaloStage2TowerAlgorithmFirmwareImp1::CaloStage2TowerAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{

}


l1t::CaloStage2TowerAlgorithmFirmwareImp1::~CaloStage2TowerAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2TowerAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & inTowers,
							      std::vector<l1t::CaloTower> & outTowers) {


  outTowers = inTowers;

}
