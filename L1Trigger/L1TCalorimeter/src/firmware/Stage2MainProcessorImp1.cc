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
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxEGAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxTauAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxJetAlgoFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxSumsAlgoFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

using namespace std;

l1t::Stage2MainProcessorFirmwareImp1::Stage2MainProcessorFirmwareImp1(unsigned fwv, CaloParamsHelper const* params) {
  m_towerAlgo = make_unique<Stage2TowerDecompressAlgorithmFirmwareImp1>(params);
  m_egClusterAlgo = make_unique<Stage2Layer2ClusterAlgorithmFirmwareImp1>(
      params, Stage2Layer2ClusterAlgorithmFirmwareImp1::ClusterInput::EH);
  m_egAlgo = make_unique<Stage2Layer2EGammaAlgorithmFirmwareImp1>(params);
  m_tauClusterAlgo = make_unique<Stage2Layer2ClusterAlgorithmFirmwareImp1>(
      params, Stage2Layer2ClusterAlgorithmFirmwareImp1::ClusterInput::EH);
  m_tauAlgo = make_unique<Stage2Layer2TauAlgorithmFirmwareImp1>(params);
  m_jetAlgo = make_unique<Stage2Layer2JetAlgorithmFirmwareImp1>(params);
  m_sumAlgo = make_unique<Stage2Layer2EtSumAlgorithmFirmwareImp1>(params);
  m_jetSumAlgo = make_unique<Stage2Layer2JetSumAlgorithmFirmwareImp1>(params);

  m_demuxEGAlgo = make_unique<Stage2Layer2DemuxEGAlgoFirmwareImp1>(params);
  m_demuxTauAlgo = make_unique<Stage2Layer2DemuxTauAlgoFirmwareImp1>(params);
  m_demuxJetAlgo = make_unique<Stage2Layer2DemuxJetAlgoFirmwareImp1>(params);
  m_demuxSumsAlgo = make_unique<Stage2Layer2DemuxSumsAlgoFirmwareImp1>(params);
}

//need to switch to BXVector
void l1t::Stage2MainProcessorFirmwareImp1::processEvent(const std::vector<l1t::CaloTower>& inTowers,
                                                        std::vector<l1t::CaloTower>& outTowers,
                                                        std::vector<l1t::CaloCluster>& clusters,
                                                        std::vector<l1t::EGamma>& mpEGammas,
                                                        std::vector<l1t::Tau>& mpTaus,
                                                        std::vector<l1t::Jet>& mpJets,
                                                        std::vector<l1t::EtSum>& mpSums,
                                                        std::vector<l1t::EGamma>& egammas,
                                                        std::vector<l1t::Tau>& taus,
                                                        std::vector<l1t::Jet>& jets,
                                                        std::vector<l1t::EtSum>& etSums) {
  // processing below is performed by the MP
  std::vector<l1t::CaloCluster> egClusters;
  std::vector<l1t::CaloCluster> tauClusters;
  std::vector<l1t::Jet> mpAllJets;
  std::vector<l1t::EtSum> towerSums;
  std::vector<l1t::EtSum> jetSums;

  m_towerAlgo->processEvent(inTowers, outTowers);
  m_egClusterAlgo->processEvent(outTowers, egClusters);
  m_egAlgo->processEvent(egClusters, outTowers, mpEGammas);
  m_tauClusterAlgo->processEvent(outTowers, tauClusters);
  m_tauAlgo->processEvent(tauClusters, outTowers, mpTaus);
  m_jetAlgo->processEvent(outTowers, mpJets, mpAllJets);
  m_sumAlgo->processEvent(outTowers, towerSums);
  m_jetSumAlgo->processEvent(mpAllJets, jetSums);

  clusters.insert(clusters.end(), egClusters.begin(), egClusters.end());

  mpSums.insert(mpSums.end(), towerSums.begin(), towerSums.end());
  mpSums.insert(mpSums.end(), jetSums.begin(), jetSums.end());

  // processing below is actually performed by the Demux card
  // in principle this could be done in a separate EDProduce
  // but it is done here for flexibility

  m_demuxEGAlgo->processEvent(mpEGammas, egammas);
  m_demuxTauAlgo->processEvent(mpTaus, taus);
  m_demuxJetAlgo->processEvent(mpJets, jets);
  m_demuxSumsAlgo->processEvent(mpSums, etSums);
}

void l1t::Stage2MainProcessorFirmwareImp1::print(std::ostream& out) const {
  out << "Calo Stage 2 Main Processor" << std::endl;
  out << "  Tower algo       : " << (m_towerAlgo ? 1 : 0) << std::endl;
  out << "  EG cluster algo  : " << (m_egClusterAlgo ? 1 : 0) << std::endl;
  out << "  EG ID algo       : " << (m_egAlgo ? 1 : 0) << std::endl;
  out << "  Tau cluster algo : " << (m_tauClusterAlgo ? 1 : 0) << std::endl;
  out << "  Tau ID algo      : " << (m_tauAlgo ? 1 : 0) << std::endl;
  out << "  Jet algo         : " << (m_jetAlgo ? 1 : 0) << std::endl;
  out << "  Jet sum algo     : " << (m_jetSumAlgo ? 1 : 0) << std::endl;
  out << "  Sums algo        : " << (m_sumAlgo ? 1 : 0) << std::endl;
}
