#include "L1Trigger/GlobalCaloTrigger/test/gctTestFunctions.h"

#include "FWCore/Framework/interface/ESHandle.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

#include "L1Trigger/GlobalCaloTrigger/test/gctTestElectrons.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestSingleEvent.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestUsingLhcData.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestFirmware.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestHt.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestHfEtSums.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

//=================================================================================================================
//
/// Constructor and destructor

gctTestFunctions::gctTestFunctions() :
  theElectronsTester   (new gctTestElectrons()),
  theSingleEventTester (new gctTestSingleEvent()),
  theEnergyAlgosTester (new gctTestEnergyAlgos()),
  theFirmwareTester    (new gctTestFirmware()),
  theRealDataTester    (new gctTestUsingLhcData()),
  theHtTester          (new gctTestHt()),
  theHfEtSumsTester    (new gctTestHfEtSums()),
  m_inputEmCands(), m_inputRegions(),
  m_bxStart(0), m_numOfBx(1)
{}

gctTestFunctions::~gctTestFunctions() {
  delete theElectronsTester;
  delete theEnergyAlgosTester;
  delete theFirmwareTester;
  delete theRealDataTester;
  delete theHtTester;
  delete theHfEtSumsTester;
}

//=================================================================================================================
//
/// Configuration method
void gctTestFunctions::configure(const edm::EventSetup& c)
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

  theEnergyAlgosTester->configure (chanMask.product());
  theHtTester->configure          (etScale.product(), htMissScale.product(), jfPars.product());
  theHfEtSumsTester->configure    (hfRingEtScale.product());
}

//=================================================================================================================
//
/// Clear vectors of input data
void gctTestFunctions::reset()
{
  m_inputEmCands.clear();
  m_inputRegions.clear();
  m_inputEmCands.resize(1);
  m_inputRegions.resize(1);
  m_bxStart = 0;
  m_numOfBx = 1;
}

//=================================================================================================================
//
/// Load another event into the gct. Overloaded for the various ways of doing this.
void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const bool simpleEvent, const int16_t bx)
{
  bxRangeUpdate(bx);
  m_inputRegions.at(bx-m_bxStart) = theEnergyAlgosTester->loadEvent(gct, simpleEvent, bx);
}

void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const std::string fileName, bool &endOfFile, const int16_t bx)
{
  bxRangeUpdate(bx);
  std::vector<L1CaloRegion> temp = theEnergyAlgosTester->loadEvent(gct, fileName, endOfFile, bx);
  if (endOfFile) {
    reset();
  } else {
    m_inputRegions.at(bx-m_bxStart) = temp;
  }
}

void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const std::string fileName, const int16_t bx)
{
  bxRangeUpdate(bx);
  m_inputEmCands.at(bx-m_bxStart) = theElectronsTester->loadEvent(gct, fileName, bx);
}

void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const edm::Event& iEvent, const int16_t bx)
{
  bxRangeUpdate(bx);
  m_inputRegions.at(bx-m_bxStart) = theEnergyAlgosTester->loadEvent(gct, theRealDataTester->loadEvent(iEvent, bx), bx);
}

void gctTestFunctions::loadSingleEvent(L1GlobalCaloTrigger* &gct, const std::string fileName, const int16_t bx)
{
  bxRangeUpdate(bx);
  m_inputRegions.at(bx-m_bxStart) = theEnergyAlgosTester->loadEvent(gct, theSingleEventTester->loadEvent(fileName, bx), bx);
}

//=================================================================================================================
//
/// This method is called when we are asked to process a new bunch crossing.
/// It expands the range of bunch crossings if necessary to include the new one,
/// by adding bunch crossings before or after the existing range.
/// It also calls the corresponding methods of the various testers.
void gctTestFunctions::bxRangeUpdate(const int16_t bx) {

  // If bxrel is negative we insert crossings before the current range, while
  // if it's bigger than m_numOfBx we need crossings after the current range.
  int bxRel = bx - m_bxStart;

  // Update the constants defining the range
  if (bxRel<0) {
    m_numOfBx -= bxRel;
    m_bxStart = bx;
  }
  if ( bxRel >= m_numOfBx) {
    m_numOfBx = bxRel + 1;
  }

  // Take care of inserting earlier crossings
  std::vector<L1CaloEmCand> tempEmc;
  std::vector<L1CaloRegion> tempRgn;
  for (int i=bxRel; i<0; i++) {
    m_inputEmCands.insert(m_inputEmCands.begin(), tempEmc);
    m_inputRegions.insert(m_inputRegions.begin(), tempRgn);
  }

  // Take care of inserting later crossings
  m_inputEmCands.resize(m_numOfBx);
  m_inputRegions.resize(m_numOfBx);

  // Do the same in the testers
  theEnergyAlgosTester->setBxRange(m_bxStart, m_numOfBx);
  theHtTester->setBxRange(m_bxStart, m_numOfBx);
}

//=================================================================================================================
//
/// Read the input electron data (after GCT processing).
void gctTestFunctions::fillElectronData(const L1GlobalCaloTrigger* gct)
{
  theElectronsTester->fillElectronData(gct);
}

//=================================================================================================================
//
/// Read the firmware results from a file for the next event
void gctTestFunctions::fillJetsFromFirmware(const std::string &fileName)
{
  theFirmwareTester->fillJetsFromFirmware(fileName, m_bxStart, m_numOfBx);
}

//=================================================================================================================
//
/// Read the input jet data from the jetfinders (after GCT processing).
void gctTestFunctions::fillRawJetData(const L1GlobalCaloTrigger* gct)
{
  theHtTester->fillRawJetData(gct);
}

//=================================================================================================================
//
/// Check the electron sort
bool gctTestFunctions::checkElectrons(const L1GlobalCaloTrigger* gct) const
{
  return theElectronsTester->checkElectrons(gct, m_bxStart, m_numOfBx);
}

/// Check the jet finder against results from the firmware
bool gctTestFunctions::checkJetFinder(const L1GlobalCaloTrigger* gct) const
{
  return theFirmwareTester->checkJetFinder(gct);
}

/// Check the energy sums algorithms
bool gctTestFunctions::checkEnergySums(const L1GlobalCaloTrigger* gct) const
{
  return theEnergyAlgosTester->checkEnergySums(gct);
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
bool gctTestFunctions::checkHtSums(const L1GlobalCaloTrigger* gct) const
{
  return theHtTester->checkHtSums(gct);
}

//=================================================================================================================
//
/// Check the Hf Et sums
bool gctTestFunctions::checkHfEtSums(const L1GlobalCaloTrigger* gct) const
{
  theHfEtSumsTester->reset();
  theHfEtSumsTester->fillExpectedHfSums(m_inputRegions);
  return theHfEtSumsTester->checkHfEtSums(gct, m_numOfBx);
}


//=================================================================================================================
//
/// Analyse calculation of energy sums in firmware
bool gctTestFunctions::checkEnergySumsFromFirmware(const L1GlobalCaloTrigger* gct, const std::string &fileName) const
{
  return theFirmwareTester->checkEnergySumsFromFirmware(gct, fileName, m_numOfBx);
}

//=================================================================================================================
//
/// Check against data read from hardware or a different version of the emulator
void gctTestFunctions::checkHwResults(const L1GlobalCaloTrigger* gct, const edm::Event &iEvent) const
{
  theRealDataTester->checkHwResults(gct, iEvent);
}

void gctTestFunctions::checkEmResults(const L1GlobalCaloTrigger* gct, const edm::Event &iEvent) const
{
  theRealDataTester->checkEmResults(gct, iEvent);
}
