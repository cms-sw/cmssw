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

#ifndef Stage2Layer2EtSumAlgorithmFirmware_H
#define Stage2Layer2EtSumAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EtSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2EtSumAlgorithmFirmwareImp1 : public Stage2Layer2EtSumAlgorithm {
  public:
    Stage2Layer2EtSumAlgorithmFirmwareImp1(CaloParamsHelper const* params);
    ~Stage2Layer2EtSumAlgorithmFirmwareImp1() override = default;
    void processEvent(const std::vector<l1t::CaloTower>& towers, std::vector<l1t::EtSum>& sums) override;

  private:
    CaloParamsHelper const* params_;
    int32_t towEtMetThresh_;
    int32_t towEtSumEtThresh_;
    int32_t towEtEcalSumThresh_;
    int32_t metEtaMax_;
    int32_t metEtaMaxHF_;
    int32_t ettEtaMax_;
    int32_t ettEtaMaxHF_;
    int32_t nTowThresholdHw_;
    int32_t nTowEtaMax_;
  };
}  // namespace l1t

#endif
