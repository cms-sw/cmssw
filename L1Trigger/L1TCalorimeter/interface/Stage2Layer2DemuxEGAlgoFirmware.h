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
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2DemuxEGAlgoFirmwareImp1 : public Stage2Layer2DemuxEGAlgo {
  public:
    Stage2Layer2DemuxEGAlgoFirmwareImp1(CaloParams* params); //const CaloMainProcessorParams & dbPars);
    virtual ~Stage2Layer2DemuxEGAlgoFirmwareImp1();
    virtual void processEvent(const std::vector<EGamma> & inputEgammas, 
			      std::vector<EGamma> & outputEgammas);
    
  private:

    CaloParams* params_;

  };
  
}

#endif
