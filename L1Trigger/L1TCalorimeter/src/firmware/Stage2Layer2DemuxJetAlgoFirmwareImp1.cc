///
/// \class l1t::Stage2Layer2JetAlgorithmFirmwareImp1
///
/// \author:
///
/// Description:

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxJetAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"


#include <vector>
#include <algorithm>

inline bool operator> ( l1t::Jet& a, l1t::Jet& b )
{
  if ( a.hwPt() > b.hwPt() ){
    return true;
  } else {
    return false;
  }
}

#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"

l1t::Stage2Layer2DemuxJetAlgoFirmwareImp1::Stage2Layer2DemuxJetAlgoFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
{


}


l1t::Stage2Layer2DemuxJetAlgoFirmwareImp1::~Stage2Layer2DemuxJetAlgoFirmwareImp1() {


}


void l1t::Stage2Layer2DemuxJetAlgoFirmwareImp1::processEvent(const std::vector<l1t::Jet> & inputJets,
                                                             std::vector<l1t::Jet> & outputJets) {

  // Set the output jets to the input jets
  outputJets = inputJets;

  // Sort the jets by pT
  std::vector<l1t::Jet>::iterator start(outputJets.begin());
  std::vector<l1t::Jet>::iterator end(outputJets.end());

  BitonicSort< l1t::Jet >(down,start,end);

  // Transform the eta and phi onto the ouput scales to GT
  for (std::vector<l1t::Jet>::iterator jet = outputJets.begin(); jet != outputJets.end(); ++jet )
    {

      jet->setHwPhi(2*jet->hwPhi());
      jet->setHwEta(2*jet->hwEta());

      if (jet->hwPt()>0x7FF){
        jet->setHwPt(0x7FF);
      } else {
        jet->setHwPt(jet->hwPt() & 0x7FF);
      }

    }

}
