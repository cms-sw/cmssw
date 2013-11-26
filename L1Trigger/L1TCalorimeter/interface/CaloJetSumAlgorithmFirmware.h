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

#ifndef CaloJetAlgorithmFirmware_H
#define CaloJetAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloJetAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloMainProcessorParams.h"

//#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloJetAlgorithmFirmware1 : public CaloJetAlgorithm {
  public:
    CaloJetAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloJetAlgorithmFirmware1();
    virtual void processEvent(const BXVector<Tower> & towers,
							  BXVector<Cluster> & clusters);
  private:
    CaloMainProcessorParams const & m_params;
  };
  
}

#endif