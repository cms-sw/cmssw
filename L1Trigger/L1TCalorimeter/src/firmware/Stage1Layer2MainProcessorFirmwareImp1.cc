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

using namespace std;
using namespace l1t;

// Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(/*const CaloParamsStage1 & dbPars*/
Stage1Layer2MainProcessorFirmwareImp1::Stage1Layer2MainProcessorFirmwareImp1(const int fwv, CaloParamsStage1* dbPars) : m_fwv(fwv), m_db(dbPars) {}

Stage1Layer2MainProcessorFirmwareImp1::~Stage1Layer2MainProcessorFirmwareImp1(){};

void Stage1Layer2MainProcessorFirmwareImp1::processEvent(const std::vector<CaloEmCand> & emcands,
						       const std::vector<CaloRegion> & regions,
						       std::vector<EGamma> * egammas,
						       std::vector<Tau> * taus,
						       std::vector<Jet> * jets,
						       std::vector<EtSum> * etsums){

  if (m_fwv == 1)
  { //HI algo
    m_egAlgo = new Stage1Layer2EGammaAlgorithmImpPP(m_db);
    m_sumAlgo = new Stage1Layer2CentralityAlgorithm(m_db);
    m_jetAlgo = new Stage1Layer2JetAlgorithmImpHI(m_db); //fwv =1 => HI algo
    m_tauAlgo = new Stage1Layer2SingleTrackHI(m_db); //fwv=1 => single track seed
  }
  else if( m_fwv == 2 )
  { //PP algorithm
    m_egAlgo = new Stage1Layer2EGammaAlgorithmImpPP(m_db);
    m_sumAlgo = new Stage1Layer2EtSumAlgorithmImpPP(m_db);
    m_jetAlgo = new Stage1Layer2JetAlgorithmImpPP(m_db); //fwv =2 => PP algo
    m_tauAlgo = new Stage1Layer2TauAlgorithmImpPP(m_db);
  }
  else{ // undefined fwv version
    edm::LogError("FWVersionError")
      << "Undefined firmware version passed to Stage1Layer2MainProcessorFirmwareImp1" << std::endl;
    return;
  }

  m_jetAlgo->processEvent(regions, emcands, jets); // need to run jets before egammas and taus for rel. isol. cuts
  m_egAlgo->processEvent(emcands, regions, jets, egammas);
  m_tauAlgo->processEvent(emcands, regions, jets, taus);
  m_sumAlgo->processEvent(regions, emcands, etsums);

  delete m_jetAlgo;
  delete m_egAlgo;
  delete m_tauAlgo;
  delete m_sumAlgo;
}
