#include <iostream>
#include "L1Trigger/GlobalCaloTrigger/test/L1GctTest.h"

#include "L1Trigger/GlobalCaloTrigger/test/gctTestFunctions.h"

#include "FWCore/Framework/interface/ESHandle.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHtMissLut.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctTest::L1GctTest(const edm::ParameterSet& iConfig) :
  theElectronTestIsEnabled   (iConfig.getUntrackedParameter<bool>("doElectrons",   false)),
  theSingleEventTestIsEnabled(iConfig.getUntrackedParameter<bool>("doSingleEvent", false)),
  theEnergyAlgosTestIsEnabled(iConfig.getUntrackedParameter<bool>("doEnergyAlgos", false)),
  theFirmwareTestIsEnabled   (iConfig.getUntrackedParameter<bool>("doFirmware",    false)),
  theRealDataTestIsEnabled   (iConfig.getUntrackedParameter<bool>("doRealData",    false)),
  theUseNewTauAlgoFlag       (iConfig.getUntrackedParameter<bool>("useNewTauAlgo", false)),
  theConfigParamsPrintFlag   (iConfig.getUntrackedParameter<bool>("printConfig",   false)),
  theInputDataFileName       (iConfig.getUntrackedParameter<std::string>("inputFile",     "")),
  theReferenceDataFileName   (iConfig.getUntrackedParameter<std::string>("referenceFile", "")),
  theEnergySumsDataFileName  (iConfig.getUntrackedParameter<std::string>("energySumsFile", "")),
  m_firstBx (-iConfig.getParameter<unsigned>("preSamples")),
  m_lastBx  ( iConfig.getParameter<unsigned>("postSamples")),
  m_eventNo(0), m_allGood(true)
{
  //now do what ever initialization is needed
  // check the files are specified if required
  if (theElectronTestIsEnabled && theInputDataFileName=="") {
    throw cms::Exception ("L1GctTestInitialisationError")
      << "no input filename provided for electron tests.\n"
      << "Specify non-blank parameter inputFile in cmsRun configuration\n"; }
  if (theFirmwareTestIsEnabled && theInputDataFileName=="") {
    throw cms::Exception ("L1GctTestInitialisationError")
      << "no input filename provided for firmware tests.\n" 
      << "Specify non-blank parameter inputFile in cmsRun configuration\n"; }
  if (theSingleEventTestIsEnabled && theInputDataFileName=="") {
    throw cms::Exception ("L1GctTestInitialisationError")
      << "no input filename provided for single event tests.\n" 
      << "Specify non-blank parameter inputFile in cmsRun configuration\n"; }
  if (theFirmwareTestIsEnabled && theReferenceDataFileName=="") {
    throw cms::Exception ("L1GctTestInitialisationError")
      << "no reference filename provided for firmware tests.\n"
      << "Specify non-blank parameter referenceFile in cmsRun configuration\n"; }
}


L1GctTest::~L1GctTest()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

   delete m_gct;
   delete m_tester;

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1GctTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;

   bool endOfFile=false;

   configureGct(iSetup);
   m_tester->configure(iSetup);

   m_gct->setupTauAlgo(theUseNewTauAlgoFlag, false);
   if (theConfigParamsPrintFlag) configParamsPrint(std::cout);

   // Initialise the gct
   m_gct->reset();
   m_tester->reset();

   for (int bx=m_firstBx; bx<=m_lastBx; bx++) {
     // Load data into the gct according to the tests to be carried out
     if (theElectronTestIsEnabled) {
       m_tester->loadNextEvent(m_gct,theInputDataFileName, bx); }

     if (theEnergyAlgosTestIsEnabled) {
       m_tester->loadNextEvent(m_gct, (100>m_eventNo), bx); }

     if (theFirmwareTestIsEnabled) {
       m_tester->loadNextEvent(m_gct, theInputDataFileName, endOfFile, bx);
       if (endOfFile) break; }

     if (theRealDataTestIsEnabled) {
       m_tester->loadNextEvent(m_gct, iEvent, bx); }

     if (theSingleEventTestIsEnabled) {
       m_tester->loadSingleEvent(m_gct, theInputDataFileName, bx); }
   }

   // Run the gct emulator on the input data
   m_gct->process();

   bool passAllTests = true;

   // Check the results of the emulator
   if (theElectronTestIsEnabled) {
     m_tester->fillElectronData(m_gct);
     passAllTests &= m_tester->checkElectrons(m_gct);
   }

   if (theFirmwareTestIsEnabled && !endOfFile) {
     m_tester->fillJetsFromFirmware(theReferenceDataFileName);
     passAllTests &= m_tester->checkJetFinder(m_gct);
     passAllTests &= m_tester->checkEnergySumsFromFirmware(m_gct, theEnergySumsDataFileName);
   }

   if (theEnergyAlgosTestIsEnabled || theSingleEventTestIsEnabled || theRealDataTestIsEnabled || (theFirmwareTestIsEnabled && !endOfFile)) {
     m_tester->fillRawJetData(m_gct);
     passAllTests &= m_tester->checkEnergySums(m_gct);
     passAllTests &= m_tester->checkHtSums(m_gct);
     passAllTests &= m_tester->checkHfEtSums(m_gct);
   }

   if (theRealDataTestIsEnabled) {
     m_tester->checkHwResults(m_gct, iEvent);
     m_tester->checkEmResults(m_gct, iEvent);
   }

   m_eventNo++;
   if (theFirmwareTestIsEnabled && endOfFile) {
     edm::LogInfo("L1GctTest") << "Reached the end of input file after " << m_eventNo << " events\n";
   }
   theFirmwareTestIsEnabled &= !endOfFile;

   // bale out if we fail any test
   m_allGood &= passAllTests;
   if (passAllTests)
   {
      //edm::LogInfo("L1GctTest") << "All tests passed for this event!" << std::endl;
   } else {
      throw cms::Exception("L1GctTestError") << "\ntest failed\n\n";
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1GctTest::beginJob()
{

  // instantiate the GCT
  m_gct = new L1GlobalCaloTrigger(L1GctJetLeafCard::hardwareJetFinder);
  m_tester = new gctTestFunctions();

  // Fill the jetEtCalibLuts vector
  lutPtr nextLut( new L1GctJetEtCalibrationLut() );

  for (unsigned ieta=0; ieta<L1GctJetFinderBase::COL_OFFSET; ieta++) {
    nextLut->setEtaBin(ieta);
    m_jetEtCalibLuts.push_back(nextLut);
    nextLut.reset ( new L1GctJetEtCalibrationLut() );
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GctTest::endJob() {
  if (m_allGood) {
    edm::LogInfo("L1GctTest") << "\n\n=== All tests passed Ok! ===\n\n" << std::endl;
  } else {
    edm::LogInfo("L1GctTest") << "\n\n=== Tests unsuccessful, exiting after "
              << m_eventNo << " events ===\n\n" << std::endl;
  }
}

void 
L1GctTest::configureGct(const edm::EventSetup& c)
{
  // get data from EventSetup
  edm::ESHandle< L1GctJetFinderParams > jfPars ;
  c.get< L1GctJetFinderParamsRcd >().get( jfPars ) ; // which record?
  edm::ESHandle< L1GctChannelMask > chanMask ;
  c.get< L1GctChannelMaskRcd >().get( chanMask ) ; // which record?
  edm::ESHandle< L1CaloEtScale > etScale ;
  c.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?
  edm::ESHandle< L1CaloEtScale > htMissScale ;
  c.get< L1HtMissScaleRcd >().get( htMissScale ) ; // which record?
  edm::ESHandle< L1CaloEtScale > hfRingEtScale ;
  c.get< L1HfRingEtScaleRcd >().get( hfRingEtScale ) ; // which record?

  m_gct->setJetFinderParams(jfPars.product());

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
  m_gct->setChannelMask(chanMask.product());
}

void L1GctTest::configParamsPrint(std::ostream & out)
{
  out << "Printing configuration parameters" << std::endl;
  out << *m_gct->getJetFinderParams();
  out << "LSB for region Et is " << m_gct->getJetFinderParams()->getRgnEtLsbGeV() << "; LSB for Ht is " << m_gct->getJetFinderParams()->getHtLsbGeV() << std::endl;
  out << "Jet seed is " << m_gct->getJetFinderParams()->getCenJetEtSeedGeV() << " GeV; or " << m_gct->getJetFinderParams()->getCenJetEtSeedGct() << " GCT units" << std::endl;
  out << "Tau isolation threshold is " << m_gct->getJetFinderParams()->getTauIsoEtThresholdGeV()
      << " GeV; or " << m_gct->getJetFinderParams()->getTauIsoEtThresholdGct() << " GCT units" << std::endl;
  out << "Jet threshold for HTT is " << m_gct->getJetFinderParams()->getHtJetEtThresholdGeV()
      << " GeV; or " << m_gct->getJetFinderParams()->getHtJetEtThresholdGct() << " GCT units" << std::endl;
  out << "Jet threshold for HTM is " << m_gct->getJetFinderParams()->getMHtJetEtThresholdGeV()
      << " GeV; or " << m_gct->getJetFinderParams()->getMHtJetEtThresholdGct() << " GCT units" << std::endl;
  out << "HtMiss Lut details: " << *m_gct->getEnergyFinalStage()->getHtMissLut()->etScale() << std::endl;
}
