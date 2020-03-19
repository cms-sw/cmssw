///
/// Description: Firmware headers
///
/// Implementation:
/// Collects concrete algorithm implmentations.
///
/// \author: R. Alex Barbieri MIT
///

//
// This header file contains the class definitions for all of the concrete
// implementations of the firmware interface. The Stage1Layer2FirmwareFactory
// selects the appropriate implementation based on the firmware version in the
// configuration.
//

#ifndef L1TCALOSTAGE1RingSUMSALGORITHMIMP_H
#define L1TCALOSTAGE1RingSUMSALGORITHMIMP_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFRingSumAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  class Stage1Layer2FlowAlgorithm : public Stage1Layer2HFRingSumAlgorithm {
  public:
    Stage1Layer2FlowAlgorithm(CaloParamsHelper const* params);
    ~Stage1Layer2FlowAlgorithm() override = default;
    void processEvent(const std::vector<l1t::CaloRegion>& regions,
                      const std::vector<l1t::CaloEmCand>& EMCands,
                      const std::vector<l1t::Tau>* taus,
                      l1t::CaloSpare* spare) override;

  private:
    std::vector<double> cosPhi;
    std::vector<double> sinPhi;
  };

  class Stage1Layer2CentralityAlgorithm : public Stage1Layer2HFRingSumAlgorithm {
  public:
    Stage1Layer2CentralityAlgorithm(CaloParamsHelper const* params);
    ~Stage1Layer2CentralityAlgorithm() override = default;
    void processEvent(const std::vector<l1t::CaloRegion>& regions,
                      const std::vector<l1t::CaloEmCand>& EMCands,
                      const std::vector<l1t::Tau>* taus,
                      l1t::CaloSpare* spare) override;

  private:
    CaloParamsHelper const* params_;
  };

  class Stage1Layer2DiTauAlgorithm : public Stage1Layer2HFRingSumAlgorithm {
  public:
    Stage1Layer2DiTauAlgorithm(CaloParamsHelper const* params);
    ~Stage1Layer2DiTauAlgorithm() override = default;
    void processEvent(const std::vector<l1t::CaloRegion>& regions,
                      const std::vector<l1t::CaloEmCand>& EMCands,
                      const std::vector<l1t::Tau>* taus,
                      l1t::CaloSpare* spare) override;

  private:
    CaloParamsHelper const* params_;
  };
}  // namespace l1t

#endif
