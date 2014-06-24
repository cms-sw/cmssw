///
/// \class l1t::Stage2Layer2MainProcessorFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 processing

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2MainProcessorFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerDecompressAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2ClusterAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EGammaAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EtSumAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetSumAlgorithmFirmware.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace std;

l1t::Stage2MainProcessorFirmwareImp1::Stage2MainProcessorFirmwareImp1(unsigned fwv, CaloParams* params) :
  m_fwv(fwv),
  m_params(params)
{

  m_towerAlgo = new Stage2TowerDecompressAlgorithmFirmwareImp1(m_params);
  m_egClusterAlgo = new Stage2Layer2ClusterAlgorithmFirmwareImp1(m_params,
							       Stage2Layer2ClusterAlgorithmFirmwareImp1::ClusterInput::E);
  m_egAlgo = new Stage2Layer2EGammaAlgorithmFirmwareImp1(m_params);
  m_tauClusterAlgo = new Stage2Layer2ClusterAlgorithmFirmwareImp1(m_params, 
								Stage2Layer2ClusterAlgorithmFirmwareImp1::ClusterInput::EH);
  dynamic_cast<Stage2Layer2ClusterAlgorithmFirmwareImp1*>(m_tauClusterAlgo)->trimCorners(false); // maybe have to think to a better solution without need to dynamic cast
  m_tauAlgo = new Stage2Layer2TauAlgorithmFirmwareImp1(m_params);
  m_jetAlgo = new Stage2Layer2JetAlgorithmFirmwareImp1(m_params);
  m_sumAlgo = new Stage2Layer2EtSumAlgorithmFirmwareImp1(m_params);
  m_jetSumAlgo = new Stage2Layer2JetSumAlgorithmFirmwareImp1(m_params);
  
}

l1t::Stage2MainProcessorFirmwareImp1::~Stage2MainProcessorFirmwareImp1()
{ 

};


//need to switch to BXVector
void l1t::Stage2MainProcessorFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & inTowers,
							std::vector<l1t::CaloTower> & outTowers,
							std::vector<l1t::CaloCluster> & clusters,
							std::vector<l1t::EGamma> & egammas,
							std::vector<l1t::Tau> & taus,
							std::vector<l1t::Jet> & jets,
							std::vector<l1t::EtSum> & etSums) {

  std::vector<l1t::CaloCluster> egClusters;
  std::vector<l1t::CaloCluster> tauClusters;
  std::vector<l1t::EtSum> towerSums;
  std::vector<l1t::EtSum> jetSums;
  
  m_towerAlgo->processEvent( inTowers, outTowers );
  m_egClusterAlgo->processEvent( outTowers, egClusters );
  m_egAlgo->processEvent( egClusters, outTowers, egammas );
  m_tauClusterAlgo->processEvent( outTowers, tauClusters );
  m_tauAlgo->processEvent( tauClusters, taus );
  m_jetAlgo->processEvent( outTowers, jets );
  m_sumAlgo->processEvent( outTowers, towerSums );
  m_jetSumAlgo->processEvent( jets, jetSums );  

  clusters.insert( clusters.end(), egClusters.begin(), egClusters.end() );

  etSums.insert( etSums.end(), towerSums.begin(), towerSums.end() );
  etSums.insert( etSums.end(), jetSums.begin(), jetSums.end() );

}


void l1t::Stage2MainProcessorFirmwareImp1::print(std::ostream& out) const {

  out << "Calo Stage 2 Main Processor" << std::endl;
  out << "  Tower algo       : " << (m_towerAlgo?1:0) << std::endl;
  out << "  EG cluster algo  : " << (m_egClusterAlgo?1:0) << std::endl;
  out << "  EG ID algo       : " << (m_egAlgo?1:0) << std::endl;
  out << "  Tau cluster algo : " << (m_tauClusterAlgo?1:0) << std::endl;
  out << "  Tau ID algo      : " << (m_tauAlgo?1:0) << std::endl;
  out << "  Jet algo         : " << (m_jetAlgo?1:0) << std::endl;
  out << "  Jet sum algo     : " << (m_jetSumAlgo?1:0) << std::endl;
  out << "  Sums algo        : " << (m_sumAlgo?1:0) << std::endl;

}


