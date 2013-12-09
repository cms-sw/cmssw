///
/// \class l1t::CaloStage1MainProcessorFirmwareImp1
///
///
/// \author: R. Alex Barbieri MIT
///

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessorFirmware.h"

#include "CaloStage1JetAlgorithmImp.h"

using namespace std;
using namespace l1t;

CaloStage1MainProcessorFirmwareImp1::CaloStage1MainProcessorFirmwareImp1(/*const CaloParams & dbPars*/)/* : m_db(dbPars)*/ {}

CaloStage1MainProcessorFirmwareImp1::~CaloStage1MainProcessorFirmwareImp1(){};

//need to switch to BXVector
void CaloStage1MainProcessorFirmwareImp1::processEvent(const EcalTriggerPrimitiveDigiCollection &,
						       const HcalTriggerPrimitiveCollection &,
						       BXVector<EGamma> & egammas,
						       BXVector<Tau> & taus,
						       BXVector<Jet> & jets,
						       BXVector<EtSum> & etsums){

  //const std::vector<l1t::CaloRegion> & regions,
  //std::vector<l1t::Jet> & jets){

  //if (db.firmwareVersion() == 1) {
  m_jetAlgo = new CaloStage1JetAlgorithmImpHI(/*m_db*/); //fwv =1 => HI algo
    //} else {
    //}

  //need to convert EcalTriggerPrimitiveDigiCollection to regions

  processEvent(regions, jets);

}
