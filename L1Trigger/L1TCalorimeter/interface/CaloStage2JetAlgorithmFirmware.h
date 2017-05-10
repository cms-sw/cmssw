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

#ifndef CaloStage2JetAlgorithmFirmware_H
#define CaloStage2JetAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2JetAlgorithmFirmware1 : public CaloStage2JetAlgorithm {
  public:
    CaloStage2JetAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloStage2JetAlgorithmFirmware1();
    virtual void processEvent(const std::vector<Tower> & towers,
			      std::vector<Cluster> & clusters);
  private:
    CaloParams const & m_params;
  };
  
}

#endif
