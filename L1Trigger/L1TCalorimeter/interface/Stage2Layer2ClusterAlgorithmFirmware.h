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

#ifndef Stage2Layer2ClusterAlgorithmFirmware_H
#define Stage2Layer2ClusterAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2ClusterAlgorithm.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2ClusterAlgorithmFirmwareImp1 : public Stage2Layer2ClusterAlgorithm {
  public:
    enum ClusterInput{
      E  = 0,
      H  = 1,
      EH = 2
    };

    Stage2Layer2ClusterAlgorithmFirmwareImp1(CaloParams* params, ClusterInput clusterInput);
    virtual ~Stage2Layer2ClusterAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::CaloCluster> & clusters);
    //
    void trimCorners(bool trimCorners=true);

  private:
    void clustering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);
    void filtering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);
    void sharing(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);
    void refining(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters);

    // parameters
    ClusterInput m_clusterInput;
    int m_seedThreshold;
    int m_clusterThreshold;
    int m_hcalThreshold;
    bool m_trimCorners;
    CaloParams* params_;
  };
  
}

#endif
