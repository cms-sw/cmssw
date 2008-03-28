#include "L1Trigger/GlobalCaloTrigger/test/L1GctTest.h"

#include "L1Trigger/GlobalCaloTrigger/test/gctTestFunctions.h"

#include "FWCore/Framework/interface/ESHandle.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

using std::cout;
using std::endl;

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
  theEnergyAlgosTestIsEnabled(iConfig.getUntrackedParameter<bool>("doEnergyAlgos", false)),
  theFirmwareTestIsEnabled   (iConfig.getUntrackedParameter<bool>("doFirmware",    false)),
  theInputDataFileName       (iConfig.getUntrackedParameter<std::string>("inputFile",     "")),
  theReferenceDataFileName   (iConfig.getUntrackedParameter<std::string>("referenceFile", "")),
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
  if (theFirmwareTestIsEnabled && theReferenceDataFileName=="") {
    throw cms::Exception ("L1GctTestInitialisationError")
      << "no reference filename provided for firmware tests.\n"
      << "Specify non-blank parameter referenceFile in cmsRun configuration\n"; }

  // instantiate the GCT
  m_gct = new L1GlobalCaloTrigger(L1GctJetLeafCard::hardwareJetFinder);
  m_tester = new gctTestFunctions();

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

   std::cout << "Hello from gctTest::analyze()" << std::endl;
  bool endOfFile=false;

  configureGct(iSetup);
   std::cout << "Done configure" << std::endl;

   // Initialise the gct
   m_gct->reset();
   m_tester->reset();
   std::cout << "Done reset" << std::endl;

   for (int bx=m_firstBx; bx<=m_lastBx; bx++) {
     // Load data into the gct according to the tests to be carried out
     if (theElectronTestIsEnabled) {
       m_tester->loadNextEvent(m_gct,theInputDataFileName, bx); }

     if (theEnergyAlgosTestIsEnabled) {
       m_tester->loadNextEvent(m_gct, (100>m_eventNo), bx); }

     if (theFirmwareTestIsEnabled) {
       m_tester->loadNextEvent(m_gct, theInputDataFileName, endOfFile, bx);
       if (endOfFile) break; }
   }
   std::cout << "Done load events" << std::endl;

   // Run the gct emulator on the input data
   m_gct->process();
   std::cout << "Done process events" << std::endl;

   bool passAllTests = true;

   // Check the results of the emulator
   if (theElectronTestIsEnabled) {
     m_tester->fillElectronData(m_gct);
     passAllTests &= m_tester->checkElectrons(m_gct);
   }
   std::cout << "Done check electrons" << std::endl;

   if (theFirmwareTestIsEnabled && !endOfFile) {
     m_tester->fillJetsFromFirmware(theReferenceDataFileName);
     passAllTests &= m_tester->checkJetFinder(m_gct);
   }
   std::cout << "Done check jet finder" << std::endl;

   if (theEnergyAlgosTestIsEnabled || (theFirmwareTestIsEnabled && !endOfFile)) {
     m_tester->fillRawJetData(m_gct);
     passAllTests &= m_tester->checkEnergySums(m_gct);
     std::cout << "Done check energy sums" << std::endl;
     passAllTests &= m_tester->checkHtSums(m_gct);
     std::cout << "Done check ht sums" << std::endl;
     passAllTests &= m_tester->checkJetCounts(m_gct);
     std::cout << "Done check jet counts" << std::endl;
     passAllTests &= m_tester->checkHfEtSums(m_gct);
     std::cout << "Done check hf et sums" << std::endl;
   }
   std::cout << "Done check events" << std::endl;

   m_eventNo++;
   if (theFirmwareTestIsEnabled && endOfFile) {
     std::cout << "Reached the end of input file after " << m_eventNo << " events\n";
   }
   theFirmwareTestIsEnabled &= !endOfFile;

   // bale out if we fail any test
   m_allGood &= passAllTests;
   if (passAllTests)
   {
      //std::cout << "All tests passed for this event!" << std::endl;
   } else {
      throw cms::Exception("L1GctTestError") << "\ntest failed\n\n";
   }
   std::cout << "Done this event" << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1GctTest::beginJob(const edm::EventSetup& c)
{
  configureGct(c);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GctTest::endJob() {
  if (m_allGood) {
    std::cout << "\n\n=== All tests passed Ok! ===\n\n" << std::endl;
  } else {
    std::cout << "\n\n=== Tests unsuccessful, exiting after "
              << m_eventNo << " events ===\n\n" << std::endl;
  }
}

void 
L1GctTest::configureGct(const edm::EventSetup& c)
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
  edm::ESHandle< L1CaloEtScale > etScale ;
  c.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?

  if (calibFun.product() == 0) {
    throw cms::Exception("L1GctConfigError")
      << "Failed to find a L1GctJetCalibFunRcd:L1GctJetEtCalibrationFunction in EventSetup!" << endl
      << "Cannot continue without this function" << endl;
  }

  m_gct->setJetFinderParams(jfPars.product());

  // make a jet Et Lut and tell it about the scales
  m_jetEtCalibLut = new L1GctJetEtCalibrationLut();

  m_jetEtCalibLut->setFunction(calibFun.product());
  m_jetEtCalibLut->setOutputEtScale(etScale.product());

  m_gct->setJetEtCalibrationLut(m_jetEtCalibLut);
  m_gct->setupJetCounterLuts(jcPosPars.product(), jcNegPars.product());

}

