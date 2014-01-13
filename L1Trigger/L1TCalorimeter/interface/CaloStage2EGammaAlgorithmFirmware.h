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
//#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2EGammaAlgorithmFirmwareImp1 : public CaloStage2EGammaAlgorithm {
  public:
    CaloStage2EGammaAlgorithmFirmwareImp1(); //const CaloMainProcessorParams & dbPars);
    virtual ~CaloStage2EGammaAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<CaloCluster> & clusters,
			      std::vector<EGamma> & egammas);
    
  private:
    //    CaloParams const & m_params;
  };
  
}

#endif
