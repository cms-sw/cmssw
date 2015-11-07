///
/// \class l1t::Stage2Layer2EGAlgorithmFirmwareImp1
///
/// \author:
///
/// Description:

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxEGAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include <vector>
#include <algorithm>

l1t::Stage2Layer2DemuxEGAlgoFirmwareImp1::Stage2Layer2DemuxEGAlgoFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxEGAlgoFirmwareImp1::~Stage2Layer2DemuxEGAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxEGAlgoFirmwareImp1::processEvent(const std::vector<l1t::EGamma> & inputEGammas,
    std::vector<l1t::EGamma> & outputEGammas) {


  outputEGammas = inputEGammas;

}
