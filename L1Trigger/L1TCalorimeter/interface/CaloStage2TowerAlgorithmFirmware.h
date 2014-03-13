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

#ifndef CaloStage2TowerAlgorithmFirmware_H
#define CaloStage2TowerAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TowerAlgorithm.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2TowerAlgorithmFirmwareImp1 : public CaloStage2TowerAlgorithm {
  public:
    CaloStage2TowerAlgorithmFirmwareImp1(CaloParams* params);
    virtual ~CaloStage2TowerAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers);
  private:
    CaloParams* params_;

  };
  
}

#endif
