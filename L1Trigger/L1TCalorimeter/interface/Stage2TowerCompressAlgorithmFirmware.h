///
/// Description: Firmware headers
///
/// Implementation:
///    Concrete firmware implementations
///
/// \author: Jim Brooke - University of Bristol
///

//
//

#ifndef Stage2TowerCompressAlgorithmFirmware_H
#define Stage2TowerCompressAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerCompressAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2TowerCompressAlgorithmFirmwareImp1 : public Stage2TowerCompressAlgorithm {
  public:
    Stage2TowerCompressAlgorithmFirmwareImp1(CaloParamsHelper const* params);
    ~Stage2TowerCompressAlgorithmFirmwareImp1() override;
    void processEvent(const std::vector<l1t::CaloTower>& inTowers, std::vector<l1t::CaloTower>& outTowers) override;

  private:
    CaloParamsHelper const* params_;
  };

}  // namespace l1t

#endif
