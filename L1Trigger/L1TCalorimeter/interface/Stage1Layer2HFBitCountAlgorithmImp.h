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

#ifndef L1TCALOSTAGE1BitCountsALGORITHMIMP_H
#define L1TCALOSTAGE1BitCountsALGORITHMIMP_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFBitCountAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"


namespace l1t {

  class Stage1Layer2HFMinimumBias : public Stage1Layer2HFBitCountAlgorithm {
  public:
    Stage1Layer2HFMinimumBias(CaloParamsHelper* params);
    virtual ~Stage1Layer2HFMinimumBias();
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      l1t::CaloSpare * spare);

  private:
    CaloParamsHelper* params_;
  };

}

#endif
