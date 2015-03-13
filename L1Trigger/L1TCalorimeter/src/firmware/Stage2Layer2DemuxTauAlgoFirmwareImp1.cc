///
/// \class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
///
/// \author: 
///
/// Description: 

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxTauAlgoFirmware.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include <vector>
#include <algorithm>

l1t::Stage2Layer2DemuxTauAlgoFirmwareImp1::Stage2Layer2DemuxTauAlgoFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxTauAlgoFirmwareImp1::~Stage2Layer2DemuxTauAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxTauAlgoFirmwareImp1::processEvent(const std::vector<l1t::Tau> & inputTaus,
    std::vector<l1t::Tau> & outputTaus) {


  outputTaus = inputTaus;

}

