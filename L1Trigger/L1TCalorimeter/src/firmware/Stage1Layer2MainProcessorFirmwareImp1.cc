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

// Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(/*const CaloParamsStage1 & dbPars*/
Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(const int fwv, CaloParamsStage1* dbPars) : m_fwv(fwv), m_db(dbPars) {
  if (m_fwv == 1)
  { //HI algo
    m_egAlgo = new Stage1Layer2EGammaAlgorithmImpPP(m_db);
    m_sumAlgo = new Stage1Layer2EtSumAlgorithmImpPP(m_db);
    m_jetAlgo = new Stage1Layer2JetAlgorithmImpHI(m_db); //fwv =1 => HI algo
    m_tauAlgo = new Stage1Layer2SingleTrackHI(m_db); //fwv=1 => single track seed
    m_hfRingAlgo = new Stage1Layer2FlowAlgorithm(m_db);
    m_hfBitAlgo = new Stage1Layer2CentralityAlgorithm(m_db);
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
    m_egAlgo = NULL;
    m_sumAlgo = NULL;
    m_tauAlgo = NULL;
    m_hfRingAlgo = NULL;
    m_hfBitAlgo = NULL;
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
							 std::vector<Jet> * jets,
							 std::vector<EtSum> * etsums,
							 std::vector<CaloSpare> * calospares){
  if(m_jetAlgo)
    m_jetAlgo->processEvent(regions, emcands, jets); // need to run jets before egammas and taus for rel. isol. cuts
  if(m_egAlgo)
  {
    // printf("%p\n",m_egAlgo);
    m_egAlgo->processEvent(emcands, regions, jets, egammas);
  }
  if(m_tauAlgo)
    m_tauAlgo->processEvent(emcands, regions, jets, taus);
  if(m_sumAlgo)
    m_sumAlgo->processEvent(regions, emcands, etsums);
  if(m_hfRingAlgo)
    m_hfRingAlgo->processEvent(regions, emcands, taus, calospares);
  if(m_hfBitAlgo)
    m_hfBitAlgo->processEvent(regions, emcands, calospares);

}
