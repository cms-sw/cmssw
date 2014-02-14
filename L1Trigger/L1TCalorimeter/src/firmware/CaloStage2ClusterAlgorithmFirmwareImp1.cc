///
/// \class l1t::CaloStage2ClusterAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2ClusterAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "



l1t::CaloStage2ClusterAlgorithmFirmwareImp1::CaloStage2ClusterAlgorithmFirmwareImp1() {
  // thresholds hard-coded for the moment
  m_seedThreshold    = 4; // 2 GeV
  m_clusterThreshold = 2; // 1 GeV

}


l1t::CaloStage2ClusterAlgorithmFirmwareImp1::~CaloStage2ClusterAlgorithmFirmwareImp1() {


}


void l1t::CaloStage2ClusterAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
							      std::vector<l1t::CaloCluster> & clusters) {



}

void l1t::CaloStage2ClusterAlgorithmFirmwareImp1::clustering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // Build clusters passing seed threshold
  for(size_t towerNr=0;towerNr<towers.size();towerNr++){
    int iEta = towers[towerNr].hwEta();
    int iPhi = towers[towerNr].hwPhi();
    int emEt = towers[towerNr].hwEtEm();
    if(emEt>m_seedThreshold)
    {
      math::XYZTLorentzVector emptyP4;
      clusters.push_back( CaloCluster(emptyP4, emEt, iEta, iPhi) );
    }
  }

  // 

}


void l1t::CaloStage2ClusterAlgorithmFirmwareImp1::filtering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
}


void l1t::CaloStage2ClusterAlgorithmFirmwareImp1::sharing(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
}

