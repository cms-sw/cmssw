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

#ifndef CaloStage1EGammaAlgorithmFirmware_H
#define CaloStage1EGammaAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1EGammaAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage1EGammaAlgorithmFirmware1 : public CaloStage1EGammaAlgorithm {
  public:
    CaloStage1EGammaAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloStage1EGammaAlgorithmFirmware1();
    virtual void processEvent(const std::vector<Tower> & towers,
			      std::vector<Cluster> & clusters);
  private:
    CaloParams const & m_params;
  };
  
}

#endif
