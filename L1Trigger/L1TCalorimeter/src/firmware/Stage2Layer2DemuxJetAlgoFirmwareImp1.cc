///
/// \class l1t::Stage2Layer2JetAlgorithmFirmwareImp1
///
/// \author: 
///
/// Description: 

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxJetAlgoFirmware.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include <vector>
#include <algorithm>

l1t::Stage2Layer2DemuxJetAlgoFirmwareImp1::Stage2Layer2DemuxJetAlgoFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxJetAlgoFirmwareImp1::~Stage2Layer2DemuxJetAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxJetAlgoFirmwareImp1::processEvent(const std::vector<l1t::Jet> & inputJets,
    std::vector<l1t::Jet> & outputJets) {


  outputJets = inputJets;

}

