#include "L1Trigger/GlobalCaloTrigger/plugins/L1GctPrintLuts.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <iostream>
#include <fstream>
#include <sys/stat.h>


L1GctPrintLuts::L1GctPrintLuts(const edm::ParameterSet& iConfig) :
  m_outputFileName(iConfig.getUntrackedParameter<std::string>("filename","gctLutContents.txt")),
  m_gct(new L1GlobalCaloTrigger(L1GctJetLeafCard::hardwareJetFinder)),
  m_jetEtCalibLut(new L1GctJetEtCalibrationLut())
{
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
}

// ------------ method called once each job just before starting event loop  ------------
// This is where we work ...
void 
L1GctPrintLuts::beginJob(const edm::EventSetup& c)
{
  // Write to a new file
  struct stat buffer ;
  if ( !stat(  m_outputFileName.c_str(), &buffer ) ) {
    edm::LogWarning("LutFileExists") << "File " << m_outputFileName << " already exists. It will not be overwritten." << std::endl; 
  } else {

    std::ofstream file;
    file.open(  m_outputFileName.c_str() );

    if (file.good()) {
      // get config data from EventSetup
      configureGct(c);
      // Print the calibration lut contents
      file << " Gct lookup table printout \n"
           << "===========================\n\n"
	   << "Jet Et Calibration lut contents\n" << std::endl;
      file << *m_jetEtCalibLut << std::endl;

      // Print the jet counter luts
      for (int wheel=0; wheel<m_gct->N_WHEEL_CARDS; wheel++) {
	// Could get the actual number of filled counters from the
	// L1GctJetCounterSetup records.
	// Just a constant for now though.
	int nCounters =  m_gct->getWheelJetFpgas().at(wheel)->N_JET_COUNTERS;
	file << "\n\n" << (wheel==0 ? "Positive " : "Negative ") << "wheel has "
	     << nCounters << " jet counters" << std::endl;
	for (int ctr=0; ctr<nCounters; ctr++) {
	  file << "\nJet counter number " << ctr << " lookup table contents \n" << std::endl;
	  file << *m_gct->getWheelJetFpgas().at(wheel)->getJetCounter(ctr)->getJetCounterLut() << std::endl;
	}
      }

    } else {
      edm::LogWarning("LutFileError") << "Error opening file " << m_outputFileName << ". No lookup tables written." << std::endl;
    }
    file.close();
  }
}

// The configuration method for the Gct - copied from L1GctEmulator
void L1GctPrintLuts::configureGct(const edm::EventSetup& c)
{
  assert(&c!=0);

  // get data from EventSetup
  edm::ESHandle< L1GctJetFinderParams > jfPars ;
  c.get< L1GctJetFinderParamsRcd >().get( jfPars ) ; // which record?
  edm::ESHandle< L1GctJetCounterSetup > jcPosPars ;
  c.get< L1GctJetCounterPositiveEtaRcd >().get( jcPosPars ) ; // which record?
  edm::ESHandle< L1GctJetCounterSetup > jcNegPars ;
  c.get< L1GctJetCounterNegativeEtaRcd >().get( jcNegPars ) ; // which record?
  edm::ESHandle< L1GctJetEtCalibrationFunction > calibFun ;
  c.get< L1GctJetCalibFunRcd >().get( calibFun ) ; // which record?
  edm::ESHandle< L1GctChannelMask > chanMask ;
  c.get< L1GctChannelMaskRcd >().get( chanMask ) ; // which record?
  edm::ESHandle< L1CaloEtScale > etScale ;
  c.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?

  if (jfPars.product() == 0) {
    throw cms::Exception("L1GctConfigError")
      << "Failed to find a L1GctJetFinderParamsRcd:L1GctJetFinderParams in EventSetup!" << std::endl
      << "Cannot continue without these parameters" << std::endl;
  }

  if (calibFun.product() == 0) {
    throw cms::Exception("L1GctConfigError")
      << "Failed to find a L1GctJetCalibFunRcd:L1GctJetEtCalibrationFunction in EventSetup!" << std::endl
      << "Cannot continue without this function" << std::endl;
  }

  if (chanMask.product() == 0) {
    throw cms::Exception("L1GctConfigError")
      << "Failed to find a L1GctChannelMaskRcd:L1GctChannelMask in EventSetup!" << std::endl
      << "Cannot continue without the channel mask" << std::endl;
  }

  m_gct->setJetFinderParams(jfPars.product());

  // tell the jet Et Lut about the scales
  m_jetEtCalibLut->setFunction(calibFun.product());
  m_jetEtCalibLut->setOutputEtScale(etScale.product());
  m_gct->setJetEtCalibrationLut(m_jetEtCalibLut);
  m_gct->setupJetCounterLuts(jcPosPars.product(), jcNegPars.product());
  m_gct->setChannelMask(chanMask.product());
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GctPrintLuts::endJob() {
}

DEFINE_ANOTHER_FWK_MODULE(L1GctPrintLuts);

