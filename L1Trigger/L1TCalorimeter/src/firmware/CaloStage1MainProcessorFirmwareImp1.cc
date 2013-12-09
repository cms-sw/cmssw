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
void CaloStage1MainProcessorFirmwareImp1::processEvent(const BXVector<CaloEmCand> & emcands,
						       const BXVector<CaloRegion> & regions,
						       BXVector<EGamma> & egammas,
						       BXVector<Tau> & taus,
						       BXVector<Jet> & jets,
						       BXVector<EtSum> & etsums){

  //if (db.firmwareVersion() == 1) {
  m_jetAlgo = new CaloStage1JetAlgorithmImpHI(/*m_db*/); //fwv =1 => HI algo
    //} else {
    //}

  //need to convert EcalTriggerPrimitiveDigiCollection to regions

  //firmware is responsible for splitting the BXVector into pieces for
  //the algos to handle
  // for(loop over BX)
  // {
  //   std::auto_ptr<std::vector<l1t::Jet>> localJets (new std::vector<l1t::Jet>);
  //   m_jetAlgo->processEvent(regions, localJets);
  std::auto_ptr<std::vector<l1t::Jet>> localJets (new std::vector<l1t::Jet>);
  std::auto_ptr<std::vector<l1t::CaloRegion>> localRegions (new std::vector<l1t::CaloRegion>);
  m_jetAlgo->processEvent(*localRegions, *localJets);

}
