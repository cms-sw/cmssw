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

#ifndef Stage2Layer2DemuxTauAlgoFirmware_H
#define Stage2Layer2DemuxTauAlgoFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxTauAlgo.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2DemuxTauAlgoFirmwareImp1 : public Stage2Layer2DemuxTauAlgo {
  public:
    Stage2Layer2DemuxTauAlgoFirmwareImp1(CaloParamsHelper* params); //const CaloMainProcessorParams & dbPars);
    virtual ~Stage2Layer2DemuxTauAlgoFirmwareImp1();
    virtual void processEvent(const std::vector<Tau> & inputTaus,
			      std::vector<Tau> & outputTaus);

  private:

    // parameters
    CaloParamsHelper* params_;

  };

}

#endif
