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
Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(const int fwv, CaloParamsHelper* dbPars) : m_fwv(fwv), m_db(dbPars) {
  if (m_fwv == 1)
  { //HI algo
    m_egAlgo = new Stage1Layer2EGammaAlgorithmImpHI(m_db);
    m_sumAlgo = new Stage1Layer2EtSumAlgorithmImpHI(m_db);
    m_jetAlgo = new Stage1Layer2JetAlgorithmImpHI(m_db); //fwv =1 => HI algo
    m_tauAlgo = new Stage1Layer2SingleTrackHI(m_db); //fwv=1 => single track seed
    m_hfRingAlgo = new Stage1Layer2CentralityAlgorithm(m_db);
    m_hfBitAlgo = new Stage1Layer2HFMinimumBias(m_db);
    // m_hfRingAlgo = new Stage1Layer2FlowAlgorithm(m_db);
    // m_hfBitAlgo = new Stage1Layer2CentralityAlgorithm(m_db);
  }
  else if( m_fwv == 2 )
  { //PP algorithm
    m_egAlgo = new Stage1Layer2EGammaAlgorithmImpPP(m_db);
    m_sumAlgo = new Stage1Layer2EtSumAlgorithmImpPP(m_db);
    m_jetAlgo = new Stage1Layer2JetAlgorithmImpPP(m_db); //fwv =2 => PP algo
    m_tauAlgo = new Stage1Layer2TauAlgorithmImpPP(m_db);
    m_hfRingAlgo = new Stage1Layer2DiTauAlgorithm(m_db);
    m_hfBitAlgo = NULL;
  }
  else if ( m_fwv == 3 )
  { // hw testing algorithms
    m_jetAlgo = new Stage1Layer2JetAlgorithmImpSimpleHW(m_db);
    m_egAlgo = new Stage1Layer2EGammaAlgorithmImpHW(m_db);
    m_sumAlgo = new Stage1Layer2EtSumAlgorithmImpHW(m_db);
    m_tauAlgo = new Stage1Layer2TauAlgorithmImpHW(m_db);
    m_hfRingAlgo = new Stage1Layer2DiTauAlgorithm(m_db);
    m_hfBitAlgo = new Stage1Layer2HFMinimumBias(m_db);
  }
  else{ // undefined fwv version
    edm::LogError("FWVersionError")
      << "Undefined firmware version passed to Stage1Layer2MainProcessorFirmwareImp1" << std::endl;
    return;
  }
}

Stage1Layer2MainProcessorFirmwareImp1::~Stage1Layer2MainProcessorFirmwareImp1(){
  delete m_jetAlgo;
  delete m_egAlgo;
  delete m_tauAlgo;
  delete m_sumAlgo;
  delete m_hfRingAlgo;
  delete m_hfBitAlgo;
};

void Stage1Layer2MainProcessorFirmwareImp1::processEvent(const std::vector<CaloEmCand> & emcands,
							 const std::vector<CaloRegion> & regions,
							 std::vector<EGamma> * egammas,
							 std::vector<Tau> * taus,
							 std::vector<Tau> * isoTaus,
							 std::vector<Jet> * jets,
							 std::vector<Jet> * preGtJets,
							 std::vector<EtSum> * etsums,
							 CaloSpare * HFringsums,
							 CaloSpare * HFbitcounts){
  if(m_jetAlgo)
    m_jetAlgo->processEvent(regions, emcands, jets, preGtJets); // need to run jets before MHT
  if(m_egAlgo)
    m_egAlgo->processEvent(emcands, regions, jets, egammas);
  if(m_tauAlgo)
    m_tauAlgo->processEvent(emcands, regions, isoTaus, taus);
  if(m_sumAlgo)
    m_sumAlgo->processEvent(regions, emcands, jets, etsums); //MHT uses jets for phi calculation
  if(m_hfRingAlgo)
    m_hfRingAlgo->processEvent(regions, emcands, isoTaus, HFringsums);
  if(m_hfBitAlgo)
    m_hfBitAlgo->processEvent(regions, emcands, HFbitcounts);

}
