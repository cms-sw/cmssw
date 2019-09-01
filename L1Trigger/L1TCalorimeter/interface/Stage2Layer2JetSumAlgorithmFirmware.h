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

#ifndef Stage2Layer2JetSumAlgorithmFirmware_H
#define Stage2Layer2JetSumAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2JetSumAlgorithmFirmwareImp1 : public Stage2Layer2JetSumAlgorithm {
  public:
    Stage2Layer2JetSumAlgorithmFirmwareImp1(CaloParamsHelper const* params);
    ~Stage2Layer2JetSumAlgorithmFirmwareImp1() override = default;
    void processEvent(const std::vector<l1t::Jet>& alljets, std::vector<l1t::EtSum>& htsums) override;

  private:
    int32_t mhtJetThresholdHw_;
    int32_t httJetThresholdHw_;
    int32_t mhtEtaMax_;
    int32_t httEtaMax_;
    int32_t mhtEtaMaxHF_;
    int32_t httEtaMaxHF_;
  };
}  // namespace l1t

#endif
