#include "L1Trigger/GlobalCaloTrigger/plugins/L1GctPrintLuts.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalHfSumAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfBitCountsLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHtMissLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfEtSumsLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <iostream>
#include <fstream>
#include <sys/stat.h>


L1GctPrintLuts::L1GctPrintLuts(const edm::ParameterSet& iConfig) :
  m_jetRanksOutFileName (iConfig.getUntrackedParameter<std::string>("jetRanksFilename", "gctJetRanksContents.txt" )),
  m_hfSumLutOutFileName (iConfig.getUntrackedParameter<std::string>("hfSumLutFilename", "gctHfSumLutContents.txt" )),
  m_htMissLutOutFileName(iConfig.getUntrackedParameter<std::string>("htMissLutFilename","gctHtMissLutContents.txt")),
  m_gct(new L1GlobalCaloTrigger(L1GctJetLeafCard::hardwareJetFinder)),
  m_jetEtCalibLuts()
{
  // Fill the jetEtCalibLuts vector
  lutPtr nextLut( new L1GctJetEtCalibrationLut() );

  for (unsigned ieta=0; ieta<L1GctJetFinderBase::COL_OFFSET; ieta++) {
    nextLut->setEtaBin(ieta);
    m_jetEtCalibLuts.push_back(nextLut);
    nextLut.reset ( new L1GctJetEtCalibrationLut() );
  }

}

L1GctPrintLuts::~L1GctPrintLuts()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1GctPrintLuts::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get config data from EventSetup
  // check this has been done successfully before proceeding
  if (configureGct(iSetup) == 0) {

    // Write to a new file
    struct stat buffer ;
    if ( !stat(  m_jetRanksOutFileName.c_str(), &buffer ) ) {
      edm::LogWarning("LutFileExists") << "File " << m_jetRanksOutFileName << " already exists. It will not be overwritten." << std::endl; 
    } else {

      std::ofstream file;
      file.open(  m_jetRanksOutFileName.c_str() );

      if (file.good()) {
	// Print the calibration lut contents
	file << " Gct lookup table printout \n"
	     << "===========================\n\n"
	     << "Jet Et Calibration lut contents\n" << std::endl;
	for (unsigned ieta=0; ieta<m_jetEtCalibLuts.size(); ieta++) {
	  file << *m_jetEtCalibLuts.at(ieta) << std::endl;
	}
      } else {
	edm::LogWarning("LutFileError") << "Error opening file " << m_jetRanksOutFileName << ". No lookup tables written." << std::endl;
      }
      file.close();
    }

    if ( !stat(  m_hfSumLutOutFileName.c_str(), &buffer ) ) {
      edm::LogWarning("LutFileExists") << "File " << m_hfSumLutOutFileName << " already exists. It will not be overwritten." << std::endl; 
    } else {

      std::ofstream file;
      file.open(  m_hfSumLutOutFileName.c_str() );

      if (file.good()) {
	// Print the Hf luts
	file << "\n\n Hf ring jet bit count luts:" << std::endl;
	file << "\n Positive eta, ring1" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getBCLut(L1GctHfEtSumsLut::bitCountPosEtaRing1) << std::endl;
	file << "\n Positive eta, ring2" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getBCLut(L1GctHfEtSumsLut::bitCountPosEtaRing2) << std::endl;
	file << "\n Negative eta, ring1" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getBCLut(L1GctHfEtSumsLut::bitCountNegEtaRing1) << std::endl;
	file << "\n Negative eta, ring2" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getBCLut(L1GctHfEtSumsLut::bitCountNegEtaRing2) << std::endl;
	file << "\n\n Hf Et sum luts:" << std::endl;
	file << "\n Positive eta, ring1" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getESLut(L1GctHfEtSumsLut::etSumPosEtaRing1) << std::endl;
	file << "\n Positive eta, ring2" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getESLut(L1GctHfEtSumsLut::etSumPosEtaRing2) << std::endl;
	file << "\n Negative eta, ring1" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getESLut(L1GctHfEtSumsLut::etSumNegEtaRing1) << std::endl;
	file << "\n Negative eta, ring2" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHfSumProcessor()->getESLut(L1GctHfEtSumsLut::etSumNegEtaRing2) << std::endl;
      } else {
	edm::LogWarning("LutFileError") << "Error opening file " << m_hfSumLutOutFileName << ". No lookup tables written." << std::endl;
      }
      file.close();
    }

    if ( !stat(  m_htMissLutOutFileName.c_str(), &buffer ) ) {
      edm::LogWarning("LutFileExists") << "File " << m_htMissLutOutFileName << " already exists. It will not be overwritten." << std::endl; 
    } else {

      std::ofstream file;
      file.open(  m_htMissLutOutFileName.c_str() );

      if (file.good()) {
	// Print the HtMiss luts
	file << "\n\n missing Ht lut:" << std::endl;
	file << *m_gct->getEnergyFinalStage()->getHtMissLut() << std::endl;
      } else {
	edm::LogWarning("LutFileError") << "Error opening file " << m_hfSumLutOutFileName << ". No lookup tables written." << std::endl;
      }
      file.close();
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
// This is where we work ...
void 
L1GctPrintLuts::beginJob()
{
}

// The configuration method for the Gct - copied from L1GctEmulator
int L1GctPrintLuts::configureGct(const edm::EventSetup& c)
{
  int success = 0;

  if (success == 0) {
    // get data from EventSetup
    edm::ESHandle< L1GctJetFinderParams > jfPars ;
    c.get< L1GctJetFinderParamsRcd >().get( jfPars ) ; // which record?
    edm::ESHandle< L1CaloEtScale > hfRingEtScale ;
    c.get< L1HfRingEtScaleRcd >().get( hfRingEtScale ) ; // which record?
    edm::ESHandle< L1CaloEtScale > htMissScale ;
    c.get< L1HtMissScaleRcd >().get( htMissScale ) ; // which record?
//     edm::ESHandle< L1GctChannelMask > chanMask ;
//     c.get< L1GctChannelMaskRcd >().get( chanMask ) ; // which record?
    edm::ESHandle< L1CaloEtScale > etScale ;
    c.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?

    if (jfPars.product() == 0) {
      success = -1;
      edm::LogWarning("L1GctConfigFailure")
	<< "Failed to find a L1GctJetFinderParamsRcd:L1GctJetFinderParams in EventSetup!" << std::endl;
    }

    if (hfRingEtScale.product() == 0) {
      success = -1;
      edm::LogWarning("L1GctConfigFailure")
	<< "Failed to find a L1HfRingEtScaleRcd:L1CaloEtScale in EventSetup!" << std::endl;
    }

    if (htMissScale.product() == 0) {
      success = -1;
      edm::LogWarning("L1GctConfigFailure")
	<< "Failed to find a L1HtMissScaleRcd:L1CaloEtScale in EventSetup!" << std::endl;
    }

//     if (chanMask.product() == 0) {
//       success = -1;
//       edm::LogWarning("L1GctConfigFailure")
// 	<< "Failed to find a L1GctChannelMaskRcd:L1GctChannelMask in EventSetup!" << std::endl;
//     }

    if (success==0) {
      // tell the jet Et Luts about the scales
      for (unsigned ieta=0; ieta<m_jetEtCalibLuts.size(); ieta++) {
	m_jetEtCalibLuts.at(ieta)->setFunction(jfPars.product());
	m_jetEtCalibLuts.at(ieta)->setOutputEtScale(etScale.product());
      }


      // pass all the setup info to the gct
      m_gct->setJetEtCalibrationLuts(m_jetEtCalibLuts);
      m_gct->setJetFinderParams(jfPars.product());
      m_gct->setHtMissScale(htMissScale.product());
      m_gct->setupHfSumLuts(hfRingEtScale.product());
      //      m_gct->setChannelMask(chanMask.product());
  
    }
  }

  if (success != 0) {
    edm::LogError("L1GctConfigError")
      << "Configuration failed - GCT emulator will not be run" << std::endl;
  }
  return success;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GctPrintLuts::endJob() {
}

DEFINE_FWK_MODULE(L1GctPrintLuts);

