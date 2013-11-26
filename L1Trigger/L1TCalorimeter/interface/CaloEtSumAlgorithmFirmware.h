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

#ifndef CaloEtSumAlgorithmFirmware_H
#define CaloEtSumAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloEtSumAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloMainProcessorParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloEtSumAlgorithmFirmware1 : public CaloEtSumAlgorithm {
  public:
    CaloEtSumAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloEtSumAlgorithmFirmware1();
    virtual void processEvent(const BXVector<Tower> & towers,
							  BXVector<Cluster> & clusters);
  private:
    CaloMainProcessorParams const & m_params;
  };
  
}

#endif