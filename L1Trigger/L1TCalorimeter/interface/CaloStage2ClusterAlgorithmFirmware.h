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

#ifndef CaloStage2ClusterAlgorithmFirmware_H
#define CaloStage2ClusterAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2ClusterAlgorithm.h"
//#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2ClusterAlgorithmFirmware1 : public CaloStage2ClusterAlgorithm {
  public:
    CaloStage2ClusterAlgorithm1(); //(const CaloParams & dbPars);
    virtual ~CaloStage2ClusterAlgorithmFirmware1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::CaloCluster> & clusters);
  private:
    //    CaloParams const & m_params;
  };
  
}

#endif
