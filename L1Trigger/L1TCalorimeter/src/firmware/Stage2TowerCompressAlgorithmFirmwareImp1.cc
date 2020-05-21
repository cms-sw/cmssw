///
/// \class l1t::Stage2TowerCompressAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerCompressAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "

l1t::Stage2TowerCompressAlgorithmFirmwareImp1::Stage2TowerCompressAlgorithmFirmwareImp1(CaloParamsHelper const* params)
    : params_(params) {}

l1t::Stage2TowerCompressAlgorithmFirmwareImp1::~Stage2TowerCompressAlgorithmFirmwareImp1() {}

void l1t::Stage2TowerCompressAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower>& inTowers,
                                                                 std::vector<l1t::CaloTower>& outTowers) {
  outTowers.reserve(outTowers.size() + inTowers.size());
  for (const auto& inTower : inTowers) {
    if (!params_->doTowerEncoding()) {
      outTowers.push_back(inTower);

    }

    else {
      int etEm = inTower.hwEtEm();
      int etHad = inTower.hwEtHad();

      int ratio = 0;
      if (etEm > 0 && etHad > 0) {
        if (etEm >= etHad)
          ratio = (int)std::round(log(float(etEm) / float(etHad)) / log(2.));
        else
          ratio = (int)std::round(log(float(etHad) / float(etEm)) / log(2.));
      }

      if (ratio >= params_->towerMaskRatio())
        ratio = params_->towerMaskRatio();

      int sum = etEm + etHad;

      // apply
      if (sum >= params_->towerMaskSum())
        sum = params_->towerMaskSum();

      int qual = 0;
      qual |= (etEm == 0 || etHad == 0 ? 0x1 : 0x0);                    // denominator ==0 flag
      qual |= ((etHad == 0 && etEm > 0) || etEm >= etHad ? 0x2 : 0x0);  // E/H flag
      qual |= (inTower.hwQual() & 0xc);                                 // get feature bits from existing tower

      l1t::CaloTower newTow;
      newTow.setHwEtEm(etEm);
      newTow.setHwEtHad(etHad);
      newTow.setHwEta(inTower.hwEta());
      newTow.setHwPhi(inTower.hwPhi());
      newTow.setHwPt(sum);
      newTow.setHwEtRatio(ratio);
      newTow.setHwQual(qual);

      outTowers.push_back(newTow);
    }
  }
}
