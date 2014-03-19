///
/// \class l1t::Stage2TowerCompressAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerCompressAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "

#include "CondFormats/L1TObjects/interface/CaloParams.h"

l1t::Stage2TowerCompressAlgorithmFirmwareImp1::Stage2TowerCompressAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{

}


l1t::Stage2TowerCompressAlgorithmFirmwareImp1::~Stage2TowerCompressAlgorithmFirmwareImp1() {


}


void l1t::Stage2TowerCompressAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & inTowers,
							      std::vector<l1t::CaloTower> & outTowers) {


  outTowers = inTowers;

}
