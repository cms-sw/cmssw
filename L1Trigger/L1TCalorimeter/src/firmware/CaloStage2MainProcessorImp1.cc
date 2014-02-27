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
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2EtSumAlgorithmFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2JetSumAlgorithmFirmware.h"

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace std;

l1t::CaloStage2MainProcessorFirmwareImp1::CaloStage2MainProcessorFirmwareImp1(const FirmwareVersion & fwv, const CaloParams & params ) :
  m_fwv(fwv),
  m_params(params)
{

  m_egClusterAlgo = new CaloStage2ClusterAlgorithmFirmwareImp1;
  m_egAlgo = new CaloStage2EGammaAlgorithmFirmwareImp1(params);
  m_tauClusterAlgo = new CaloStage2ClusterAlgorithmFirmwareImp1;
  m_tauAlgo = new CaloStage2TauAlgorithmFirmwareImp1(params);
  m_jetAlgo = new CaloStage2JetAlgorithmFirmwareImp1(params);
  m_sumAlgo = new CaloStage2EtSumAlgorithmFirmwareImp1(params);
  m_jetSumAlgo = new CaloStage2JetSumAlgorithmFirmwareImp1(params);
  
}

l1t::CaloStage2MainProcessorFirmwareImp1::~CaloStage2MainProcessorFirmwareImp1()
{ 

};

//need to switch to BXVector
void l1t::CaloStage2MainProcessorFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
						       std::vector<l1t::EGamma> & egammas,
						       std::vector<l1t::Tau> & taus,
						       std::vector<l1t::Jet> & jets,
						       std::vector<l1t::EtSum> & etsums) {

  std::vector<l1t::CaloCluster> egClusters;
  std::vector<l1t::CaloCluster> tauClusters;
  std::vector<l1t::EtSum> towersums;
  std::vector<l1t::EtSum> jetsums;
  
  m_egClusterAlgo->processEvent( towers, egClusters );
  m_egAlgo->processEvent( egClusters,towers, egammas );
  m_egClusterAlgo->processEvent( towers, tauClusters );
  m_tauAlgo->processEvent( tauClusters, taus );
  m_jetAlgo->processEvent( towers, jets );
  m_sumAlgo->processEvent( towers, towersums );
  m_jetSumAlgo->processEvent( jets, jetsums );  

  etsums.insert( etsums.end(), towersums.begin(), towersums.end() );
  etsums.insert( etsums.end(), jetsums.begin(), jetsums.end() );

}
