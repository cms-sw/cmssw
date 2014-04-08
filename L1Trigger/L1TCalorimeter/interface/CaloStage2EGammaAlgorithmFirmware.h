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

#ifndef CaloStage2EGammaAlgorithmFirmware_H
#define CaloStage2EGammaAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EGammaAlgorithm.h"
#include "L1Trigger/L1TCalorimeter/interface/LUT.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2EGammaAlgorithmFirmwareImp1 : public CaloStage2EGammaAlgorithm {
  public:
    CaloStage2EGammaAlgorithmFirmwareImp1(CaloParams* params); //const CaloMainProcessorParams & dbPars);
    virtual ~CaloStage2EGammaAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<CaloCluster> & clusters, 
			      const std::vector<CaloTower>& towers,
			      std::vector<EGamma> & egammas);
    
  private:
    CaloParams* params_;
    LUT lut_;
  };
  
}

#endif
