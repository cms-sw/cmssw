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

#ifndef Stage2PreProcessorFirmware_H
#define Stage2PreProcessorFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2PreProcessor.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerCompressAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include <memory>

namespace l1t {

  // first iteration
  class Stage2PreProcessorFirmwareImp1 : public Stage2PreProcessor {
  public:
    Stage2PreProcessorFirmwareImp1(unsigned fwv, CaloParamsHelper const* params);

    void processEvent(const std::vector<l1t::CaloTower>& inTowers, std::vector<l1t::CaloTower>& outTowers) override;

    void print(std::ostream&) const;

    friend std::ostream& operator<<(std::ostream& o, const Stage2PreProcessorFirmwareImp1& p) {
      p.print(o);
      return o;
    }

  private:
    //FirmwareVersion const & m_fwv;
    std::unique_ptr<Stage2TowerCompressAlgorithm> m_towerAlgo;
  };

}  // namespace l1t

#endif
