///
/// \class l1t::Stage2TowerCompressAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerCompressAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

l1t::Stage2TowerCompressAlgorithmFirmwareImp1::Stage2TowerCompressAlgorithmFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
{

}


l1t::Stage2TowerCompressAlgorithmFirmwareImp1::~Stage2TowerCompressAlgorithmFirmwareImp1() {


}


void l1t::Stage2TowerCompressAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & inTowers,
								 std::vector<l1t::CaloTower> & outTowers) {

  for ( auto tow = inTowers.begin();
	tow != inTowers.end();
	++tow ) {

    if (!params_->doTowerEncoding()) {

      outTowers.push_back( *tow );

    }

    else {

      int etEm  = tow->hwEtEm();
      int etHad = tow->hwEtHad();

      int ratio = 0;
      if (etEm>0 && etHad>0) {
	if (etEm>=etHad) ratio = (int) std::round(log(float(etEm) / float(etHad))/log(2.));
	else ratio = (int) std::round(log(float(etHad) / float(etEm))/log(2.));
      }
      ratio &= params_->towerMaskRatio() ;

      int sum  = etEm + etHad;
      sum &= params_->towerMaskSum() ;

      int qual = 0;
      qual |= (etEm==0 || etHad==0 ? 0x1 : 0x0 );  // denominator ==0 flag
      qual |= ((etHad==0 && etEm>0) || etEm>=etHad ? 0x2 : 0x0 );  // E/H flag
      qual |= (tow->hwQual() & 0xc); // get feature bits from existing tower

      l1t::CaloTower newTow;
      newTow.setHwEtEm(etEm);
      newTow.setHwEtHad(etHad);
      newTow.setHwEta( tow->hwEta() );
      newTow.setHwPhi( tow->hwPhi() );
      newTow.setHwPt( sum );
      newTow.setHwEtRatio( ratio );
      newTow.setHwQual( qual );

      outTowers.push_back(newTow);

    }

  }

}
