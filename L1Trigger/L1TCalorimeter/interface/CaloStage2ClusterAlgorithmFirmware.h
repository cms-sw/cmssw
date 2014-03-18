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
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class CaloStage2ClusterAlgorithmFirmwareImp1 : public CaloStage2ClusterAlgorithm {
  public:
    enum ClusterInput{
      E  = 0,
      H  = 1,
      EH = 2
    };

    CaloStage2ClusterAlgorithmFirmwareImp1(CaloParams* params, ClusterInput clusterInput);
    virtual ~CaloStage2ClusterAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::CaloCluster> & clusters);
  private:
    void clustering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);
    void filtering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);
    void sharing(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);
    void refining(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);

    // parameters
    ClusterInput m_clusterInput;
    int m_seedThreshold;
    int m_clusterThreshold;
    CaloParams* params_;
  };
  
}

#endif
