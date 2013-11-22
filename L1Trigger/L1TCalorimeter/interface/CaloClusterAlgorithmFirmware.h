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

#ifndef ClusterAlgorithmFirmware_H
#define ClusterAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/CaloClusterAlgorithm.h"
#include "CondFormats/L1TCalorimeter/interface/CaloMainProcessorParams.h"

//#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloClusterAlgorithmFirmware1 : public CaloClusterAlgorithm {
  public:
    CaloClusterAlgorithm1(const CaloMainProcessorParams & dbPars);
    virtual ~CaloClusterAlgorithmFirmware1();
    virtual void processEvent(const BXVector<Tower> & towers,
							  BXVector<Cluster> & clusters);
  private:
    CaloMainProcessorParams const & m_params;
  };
  
}

#endif