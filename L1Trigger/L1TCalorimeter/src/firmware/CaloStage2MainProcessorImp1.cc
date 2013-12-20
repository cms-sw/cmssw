///
/// \class l1t::CaloStage2MainProcessorFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 processing

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2MainProcessorFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2ClusterAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EGammaAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2TauAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EtSumFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetSumFirmware.h"

using namespace std;

CaloStage2MainProcessorFirmwareImp1::CaloStage1MainProcessorFirmwareImp1() :
  m_clusterAlgo( new CaloStage2ClusterAlgorithmFirmware1 ),
  m_egAlgo( new CaloStage2EGammaAlgorithmFirmware1 ),
  m_tauAlgo( new CaloStage2TauAlgorithmFirmware1 ),
  m_jetAlgo( new CaloStage2JetAlgorithmFirmware1 ),
  m_sumAlgo( new CaloStage2EtSumAlgorithmFirmware1 ),
  m_jetSumAlgo( new CaloStage2JetSumAlgorithmFirmware1 )
}

CaloStage2MainProcessorFirmwareImp1::~CaloStage1MainProcessorFirmwareImp1() { };

//need to switch to BXVector
void CaloStage2MainProcessorFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
						       std::vector<l1t::EGamma> & egammas,
						       std::vector<l1t::Tau> & taus,
						       std::vector<l1t::Jet> & jets,
						       std::vector<l1t::EtSum> & etsums) {

  std::vector<l1t::CaloCluster> clusters;
  std::vector<l1t::EtSum> towersums;
  std::vector<l1t::EtSum> jetsums;
  
  m_clusterAlgo.process( towers, clusters );
  m_egAlgo.process( clusters, egammas );
  m_tauAlgo.process( clusters, taus );
  m_jetAlgo.process( towers, jets );
  m_sumAlgo.process( towers, towersums );
  m_jetSumAlgo.process( jets, jetsums );  

  etsums.insert( etsums.end(), towersums.begin(), towersums.end() );
  etsums.insert( etsums.end(), jetsums.begin(), jetsums.end() );

}
