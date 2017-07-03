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

#ifndef Stage2Layer2DemuxSumsAlgoFirmware_H
#define Stage2Layer2DemuxSumsAlgoFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxSumsAlgo.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "L1Trigger/L1TCalorimeter/interface/Cordic.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2DemuxSumsAlgoFirmwareImp1 : public Stage2Layer2DemuxSumsAlgo {
  public:
    Stage2Layer2DemuxSumsAlgoFirmwareImp1(CaloParamsHelper* params);
    ~Stage2Layer2DemuxSumsAlgoFirmwareImp1() override;
    void processEvent(const std::vector<l1t::EtSum> & inputSums,
			      std::vector<l1t::EtSum> & outputSums) override;
  private:

    CaloParamsHelper* params_;

    Cordic cordic_;

  };

}

#endif
