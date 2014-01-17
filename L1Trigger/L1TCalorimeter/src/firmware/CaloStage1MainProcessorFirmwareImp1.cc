///
/// \class l1t::CaloStage1MainProcessorFirmwareImp1
///
///
/// \author: R. Alex Barbieri MIT
///
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessorFirmware.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1JetAlgorithmImp.h"

using namespace std;
using namespace l1t;

CaloStage1MainProcessorFirmwareImp1::CaloStage1MainProcessorFirmwareImp1(/*const CaloParams & dbPars*/
  const FirmwareVersion & fwv) : m_fwv(fwv)/* : m_db(dbPars)*/ {}

CaloStage1MainProcessorFirmwareImp1::~CaloStage1MainProcessorFirmwareImp1(){};

void CaloStage1MainProcessorFirmwareImp1::processEvent(const std::vector<CaloEmCand> & emcands,
						       const std::vector<CaloRegion> & regions,
						       std::vector<EGamma> & egammas,
						       std::vector<Tau> & taus,
						       std::vector<Jet> & jets,
						       std::vector<EtSum> & etsums){

  if (m_fwv.firmwareVersion() == 1) { //HI algo
    m_jetAlgo = new CaloStage1JetAlgorithmImpHI(/*m_db*/); //fwv =1 => HI algo
  } else if( m_fwv.firmwareVersion() == 2 ) { //PP algorithm
    m_jetAlgo = new CaloStage1JetAlgorithmImpPP(/*m_db*/); //fwv =2 => PP algo
  } else{ // undefined fwv version
    edm::LogError("FWVersionError")
      << "Undefined firmware version passed to CaloStage1MainProcessorFirmwareImp1" << std::endl;
    return;
  }

  m_jetAlgo->processEvent(regions, jets);
}
