///
/// \class l1t::CaloStage1MainProcessorFirmwareImp1
///
///
/// \author: R. Alex Barbieri MIT
///

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessorFirmware.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include "CaloStage1JetAlgorithmImp.h"

using namespace std;
using namespace l1t;

CaloStage1MainProcessorFirmwareImp1::CaloStage1MainProcessorFirmwareImp1(/*const CaloParams & dbPars*/
  const FirmwareVersion & fwv) : m_fwv(fwv)/* : m_db(dbPars)*/ {}

CaloStage1MainProcessorFirmwareImp1::~CaloStage1MainProcessorFirmwareImp1(){};

//need to switch to BXVector
void CaloStage1MainProcessorFirmwareImp1::processEvent(const BXVector<CaloEmCand> & emcands,
						       const BXVector<CaloRegion> & regions,
						       BXVector<EGamma> & egammas,
						       BXVector<Tau> & taus,
						       BXVector<Jet> & jets,
						       BXVector<EtSum> & etsums){

  if (m_fwv.firmwareVersion() == 1) { //HI algo
    m_jetAlgo = new CaloStage1JetAlgorithmImpHI(/*m_db*/); //fwv =1 => HI algo

    //firmware is responsible for splitting the BXVector into pieces for
    //the algos to handle
    // Hardcode bx=0 for now. TODO
    std::auto_ptr<std::vector<l1t::Jet>> localJets (new std::vector<l1t::Jet>);
    std::auto_ptr<std::vector<l1t::CaloRegion>> localRegions (new std::vector<l1t::CaloRegion>);
    for(std::vector<l1t::CaloRegion>::const_iterator region = regions.begin(0); region != regions.end(0); ++region)
    {
      localRegions->push_back(*region);
    }
    m_jetAlgo->processEvent(*localRegions, *localJets);

    for(std::vector<l1t::Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet)
    {
      jets.push_back(0, *jet);
    }
  } else if( m_fwv.firmwareVersion() == 1 ) {
    //pp algorithm should go here
  }

}
