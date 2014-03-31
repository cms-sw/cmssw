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


  for ( auto tow = inTowers.begin();
	tow != inTowers.end();
	++tow ) {

    if (!params_->doTowerCompression()) {
      
      outTowers.push_back( *tow );
      
    }
    
    else {
      
      
      int sum   = tow->hwPt();
      int ratio = tow->hwEtRatio();
      int qual  = tow->hwQual();
      
      int numCoeff = 0;
      int denomCoeff = 0;
      
      if ((qual & 0x1)==0) {
	switch (ratio) {
	case 000 : 
	  numCoeff = 64;
	  denomCoeff = 64;
	  break;
	case 001 : 
	  numCoeff = 43;
	  denomCoeff = 85;
	  break;
	case 010 : 
	  numCoeff = 26;
	  denomCoeff = 102;
	  break;
	case 011 : 
	  numCoeff = 14;
	  denomCoeff = 114;
	  break;
	case 100 : 
	  numCoeff = 8;
	  denomCoeff = 120;
	  break;
	case 101 : 
	  numCoeff = 4;
	  denomCoeff = 124;
	  break;
	case 110 : 
	  numCoeff = 2;
	  denomCoeff = 126;
	  break;
	case 111 : 
	  numCoeff = 1;
	  denomCoeff = 127;
	  break;
	}
      }
      else {
	numCoeff = 128;
	denomCoeff = 0;
      }
      
      int em    = 0;
      int had   = 0;
      
      if ((qual&0x1) != 0 && (qual&0x2) == 0) // E==0
	had = sum;
      
      if ((qual&0x1) != 0 && (qual&0x2) != 0) // H==0
	em = sum;
      
      if ((qual&0x1) == 0 && (qual&0x2) == 0) { // H > E , so ratio==log(H/E)
	em  = denomCoeff * sum;
	had = numCoeff * sum;
      }
      
      if ((qual&0x1) == 0 && (qual&0x2) != 0) { // E >= H , so ratio==log(E/H)
	em  = numCoeff * sum;
	had = denomCoeff * sum;
      }
      
      em  &= params_->towerMaskE();
      had &= params_->towerMaskH();
      
      l1t::CaloTower newTow;
      newTow.setHwEta( tow->hwEta() );
      newTow.setHwPhi( tow->hwPhi() );
      newTow.setHwEtEm( em );
      newTow.setHwEtHad( had );
      
      newTow.setHwPt( sum );
      newTow.setHwEtRatio( ratio );
      newTow.setHwQual( qual );
      
      outTowers.push_back(newTow);
      
    }
    
  }

}
