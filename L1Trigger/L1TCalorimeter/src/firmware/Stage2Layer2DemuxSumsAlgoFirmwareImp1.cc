///
/// \class l1t::Stage2Layer2SumsAlgorithmFirmwareImp1
///
/// \author: 
///
/// Description: 

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxSumsAlgoFirmware.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include <vector>
#include <algorithm>

l1t::Stage2Layer2DemuxSumsAlgoFirmwareImp1::Stage2Layer2DemuxSumsAlgoFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxSumsAlgoFirmwareImp1::~Stage2Layer2DemuxSumsAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxSumsAlgoFirmwareImp1::processEvent(const std::vector<l1t::EtSum> & inputSums,
    std::vector<l1t::EtSum> & outputSums) {


  outputSums = inputSums;

}

