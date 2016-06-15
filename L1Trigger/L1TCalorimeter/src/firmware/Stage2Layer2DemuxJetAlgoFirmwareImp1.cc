///
/// \class l1t::Stage2Layer2JetAlgorithmFirmwareImp1
///
/// \author:
///
/// Description:

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxJetAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


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

  outputJets = inputJets;

  // Sort the jets by pT
  std::vector<l1t::Jet>::iterator start(outputJets.begin());
  std::vector<l1t::Jet>::iterator end(outputJets.end());

  //  for (auto& jet: outputJets){
  //    std::cout << "MP : " << jet.hwPt() << ", " << jet.hwEta() << ", " << jet.hwPhi() << ", " << CaloTools::towerEta(jet.hwEta()) << ", " << CaloTools::towerPhi(jet.hwEta(),jet.hwPhi()) << std::endl;
  //  }

  BitonicSort< l1t::Jet >(down,start,end);

  // convert eta to GT coordinates
  for(auto& jet : outputJets){

    int gtEta = CaloTools::gtEta(jet.hwEta());
    int gtPhi = CaloTools::gtPhi(jet.hwEta(),jet.hwPhi());
    
    jet.setHwEta(gtEta);
    jet.setHwPhi(gtPhi);
    
  }
  

}
