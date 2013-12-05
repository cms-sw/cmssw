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

#ifndef CaloStage1TauAlgorithmFirmware_H
#define CaloStage1TauAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1TauAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage1TauAlgorithmFirmware1 : public CaloStage1TauAlgorithm {
  public:
    CaloStage1TauAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloStage1TauAlgorithmFirmware1();
    virtual void processEvent(const BXVector<Tower> & towers,
			      BXVector<Cluster> & clusters);
  private:
    CaloParams const & m_params;
  };
  
}

#endif
