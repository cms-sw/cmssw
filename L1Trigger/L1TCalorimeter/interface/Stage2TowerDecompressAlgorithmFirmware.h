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

#ifndef Stage2TowerDecompressAlgorithmFirmware_H
#define Stage2TowerDecompressAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerDecompressAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2TowerDecompressAlgorithmFirmwareImp1 : public Stage2TowerDecompressAlgorithm {
  public:
    Stage2TowerDecompressAlgorithmFirmwareImp1(CaloParamsHelper* params);
    virtual ~Stage2TowerDecompressAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers);
  private:
    CaloParamsHelper* params_;

  };

}

#endif
