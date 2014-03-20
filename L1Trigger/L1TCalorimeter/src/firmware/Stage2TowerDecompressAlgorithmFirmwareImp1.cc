///
/// \class l1t::CaloStage2TowerAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerDecompressAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "

#include "CondFormats/L1TObjects/interface/CaloParams.h"

l1t::Stage2TowerDecompressAlgorithmFirmwareImp1::Stage2TowerDecompressAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{

}


l1t::Stage2TowerDecompressAlgorithmFirmwareImp1::~Stage2TowerDecompressAlgorithmFirmwareImp1() {


}


void l1t::Stage2TowerDecompressAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & inTowers,
								   std::vector<l1t::CaloTower> & outTowers) {


  if (!params_->doTowerCompression()) {
    outTowers = inTowers;
    return;
  }

  for ( auto tow = inTowers.begin();
	tow != inTowers.end();
	++tow ) {

    int sum   = tow->hwPt();
    //    int ratio = tow->hwEtRatio();
    int qual  = tow->hwQual();

    int em    = 0;
    int had   = 0;

    if ((qual&0x1) != 0 && (qual&0x2) == 0) // E==0
      had = sum;
    
    if ((qual&0x1) != 0 && (qual&0x2) != 0) // H==0
      em = sum;
    
    if ((qual&0x1) == 0 && (qual&0x2) == 0) { // H > E , so ratio==log(H/E)
      em  = 1;
      had = 2;
    }
    
    if ((qual&0x1) == 0 && (qual&0x2) != 0) { // E >= H , so ratio==log(E/H)
      em  = 3;
      had = 4;
    }
    
    em  &= params_->towerMaskE();
    had &= params_->towerMaskH();

    l1t::CaloTower newTow;
    newTow.setHwEta( tow->hwEta() );
    newTow.setHwPhi( tow->hwPhi() );
    newTow.setHwEtEm( em );
    newTow.setHwEtHad( had );
    
    outTowers.push_back(newTow);

  }

}
