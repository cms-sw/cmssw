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

#ifndef CaloStage2TauAlgorithmFirmware_H
#define CaloStage2TauAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TauAlgorithm.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2TauAlgorithmFirmwareImp1 : public CaloStage2TauAlgorithm {
  public:
    CaloStage2TauAlgorithmFirmwareImp1(CaloParams* params); //const CaloMainProcessorParams & dbPars);
    virtual ~CaloStage2TauAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<CaloCluster> & clusters,
			      std::vector<Tau> & taus);
    
  private:
    CaloParams* params_;
  };
  
}

#endif
