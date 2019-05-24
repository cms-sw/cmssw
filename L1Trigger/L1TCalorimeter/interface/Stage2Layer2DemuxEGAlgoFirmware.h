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

#ifndef Stage2Layer2DemuxEGAlgoFirmware_H
#define Stage2Layer2DemuxEGAlgoFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxEGAlgo.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2DemuxEGAlgoFirmwareImp1 : public Stage2Layer2DemuxEGAlgo {
  public:
    Stage2Layer2DemuxEGAlgoFirmwareImp1(CaloParamsHelper const* params);  //const CaloMainProcessorParams & dbPars);
    ~Stage2Layer2DemuxEGAlgoFirmwareImp1() override;
    void processEvent(const std::vector<EGamma>& inputEgammas, std::vector<EGamma>& outputEgammas) override;

  private:
    CaloParamsHelper const* params_;
  };

}  // namespace l1t

#endif
