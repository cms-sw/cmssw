///
/// \class l1t::CaloStage1MainProcessorFirmwareImp1
///
///
/// \author: R. Alex Barbieri MIT
///

// This example implemenents algorithm version 1 and 2.

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessorFirmware.h"

#include "CaloStage1JetAlgorithmImp.h"

using namespace std;
using namespace l1t;

CaloStage1MainProcessorFirmwareImp1::CaloStage1MainProcessorFirmwareImp1(const CaloParams & dbPars) : m_db(dbPars) {}

CaloStage1MainProcessorFirmwareImp1::~CaloStage1MainProcessorFirmwareImp1(){};

//need to switch to BXVector
void CaloStage1MainProcessorFirmwareImp1::processEvent(const std::vector<l1t::CaloRegion> & regions,
						       std::vector<l1t::Jet> & jets){

  if (db.firmwareVersion() == 1) {
    m_jetAlgo = CaloStage1JetAlgorithmImpHI(m_db); //fwv =1 => HI algo
  } else {
  }


  processEvent(regions, jets);

}
