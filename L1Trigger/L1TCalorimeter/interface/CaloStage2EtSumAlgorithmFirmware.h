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

#ifndef CaloStage2EtSumAlgorithmFirmware_H
#define CaloStage2EtSumAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EtSumAlgorithm.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2EtSumAlgorithmFirmwareImp1 : public CaloStage2EtSumAlgorithm {
  public:
    CaloStage2EtSumAlgorithmFirmwareImp1(); //const CaloParams & dbPars);
    virtual ~CaloStage2EtSumAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::EtSum> & sums);
  private:
    //    CaloParams const & m_params;
  };
  
}

#endif
