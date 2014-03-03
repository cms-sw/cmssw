///
/// \class l1t::CaloStage2JetAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetAlgorithmFirmware.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"


l1t::CaloStage2JetAlgorithmFirmwareImp1::CaloStage2JetAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::CaloStage2JetAlgorithmFirmwareImp1::~CaloStage2JetAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2JetAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
							      std::vector<l1t::Jet> & jets) {



}

