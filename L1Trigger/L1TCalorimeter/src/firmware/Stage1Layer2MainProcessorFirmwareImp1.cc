///
/// \class l1t::Stage1Layer2MainProcessorFirmwareImp1
///
///
/// \author: R. Alex Barbieri MIT
///
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2MainProcessorFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EGammaAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EtSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2JetAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFRingSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFBitCountAlgorithmImp.h"

using namespace std;
using namespace l1t;

// Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(/*const CaloParamsHelper & dbPars*/
Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(const int fwv,
                                                                             CaloParamsHelper const* dbPars)
    : m_fwv(fwv) {
  if (m_fwv == 1) {  //HI algo
    m_egAlgo = std::make_unique<Stage1Layer2EGammaAlgorithmImpHI>(dbPars);
    m_sumAlgo = std::make_unique<Stage1Layer2EtSumAlgorithmImpHI>(dbPars);
    m_jetAlgo = std::make_unique<Stage1Layer2JetAlgorithmImpHI>(dbPars);  //fwv =1 => HI algo
    m_tauAlgo = std::make_unique<Stage1Layer2SingleTrackHI>(dbPars);      //fwv=1 => single track seed
    m_hfRingAlgo = std::make_unique<Stage1Layer2CentralityAlgorithm>(dbPars);
    m_hfBitAlgo = std::make_unique<Stage1Layer2HFMinimumBias>(dbPars);
    // m_hfRingAlgo = std::make_unique<Stage1Layer2FlowAlgorithm>(dbPars);
    // m_hfBitAlgo = std::make_unique<Stage1Layer2CentralityAlgorithm>(dbPars);
  } else if (m_fwv == 2) {  //PP algorithm
    m_egAlgo = std::make_unique<Stage1Layer2EGammaAlgorithmImpPP>(dbPars);
    m_sumAlgo = std::make_unique<Stage1Layer2EtSumAlgorithmImpPP>(dbPars);
    m_jetAlgo = std::make_unique<Stage1Layer2JetAlgorithmImpPP>(dbPars);  //fwv =2 => PP algo
    m_tauAlgo = std::make_unique<Stage1Layer2TauAlgorithmImpPP>(dbPars);
    m_hfRingAlgo = std::make_unique<Stage1Layer2DiTauAlgorithm>(dbPars);
    m_hfBitAlgo = nullptr;
  } else if (m_fwv == 3) {  // hw testing algorithms
    m_jetAlgo = std::make_unique<Stage1Layer2JetAlgorithmImpSimpleHW>(dbPars);
    m_egAlgo = std::make_unique<Stage1Layer2EGammaAlgorithmImpHW>(dbPars);
    m_sumAlgo = std::make_unique<Stage1Layer2EtSumAlgorithmImpHW>(dbPars);
    m_tauAlgo = std::make_unique<Stage1Layer2TauAlgorithmImpHW>(dbPars);
    m_hfRingAlgo = std::make_unique<Stage1Layer2DiTauAlgorithm>(dbPars);
    m_hfBitAlgo = std::make_unique<Stage1Layer2HFMinimumBias>(dbPars);
  } else {  // undefined fwv version
    edm::LogError("FWVersionError") << "Undefined firmware version passed to Stage1Layer2MainProcessorFirmwareImp1"
                                    << std::endl;
    return;
  }
}

void Stage1Layer2MainProcessorFirmwareImp1::processEvent(const std::vector<CaloEmCand>& emcands,
                                                         const std::vector<CaloRegion>& regions,
                                                         std::vector<EGamma>* egammas,
                                                         std::vector<Tau>* taus,
                                                         std::vector<Tau>* isoTaus,
                                                         std::vector<Jet>* jets,
                                                         std::vector<Jet>* preGtJets,
                                                         std::vector<EtSum>* etsums,
                                                         CaloSpare* HFringsums,
                                                         CaloSpare* HFbitcounts) {
  if (m_jetAlgo)
    m_jetAlgo->processEvent(regions, emcands, jets, preGtJets);  // need to run jets before MHT
  if (m_egAlgo)
    m_egAlgo->processEvent(emcands, regions, jets, egammas);
  if (m_tauAlgo)
    m_tauAlgo->processEvent(emcands, regions, isoTaus, taus);
  if (m_sumAlgo)
    m_sumAlgo->processEvent(regions, emcands, jets, etsums);  //MHT uses jets for phi calculation
  if (m_hfRingAlgo)
    m_hfRingAlgo->processEvent(regions, emcands, isoTaus, HFringsums);
  if (m_hfBitAlgo)
    m_hfBitAlgo->processEvent(regions, emcands, HFbitcounts);
}
