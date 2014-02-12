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

#ifndef CaloTowerAlgorithmFirmware_H
#define CaloTowerAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloTowerAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloMainProcessorParams.h"

//#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloTowerAlgorithmFirmware1 : public CaloTowerAlgorithm {
  public:
    CaloTowerAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloTowerAlgorithmFirmware1();
    virtual void processEvent(const EcalTriggerPrimitiveDigiCollection &,
							  const HcalTriggerPrimitiveCollection &,
 							  BXVector<Tower> & towers);
  private:
    CaloMainProcessorParams const & m_params;
  };
  
}

#endif