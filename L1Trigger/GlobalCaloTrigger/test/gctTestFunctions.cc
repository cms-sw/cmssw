#include "L1Trigger/GlobalCaloTrigger/test/gctTestFunctions.h"

#include "L1Trigger/GlobalCaloTrigger/test/gctTestElectrons.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestFirmware.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestHtAndJetCounts.h"
#include "L1Trigger/GlobalCaloTrigger/test/gctTestHfEtSums.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

//=================================================================================================================
//
/// Constructor and destructor

gctTestFunctions::gctTestFunctions() :
  theElectronsTester      ( new gctTestElectrons() ),
  theEnergyAlgosTester    ( new gctTestEnergyAlgos() ),
  theFirmwareTester       ( new gctTestFirmware() ),
  theHtAndJetCountsTester ( new gctTestHtAndJetCounts() ),
  theHfEtSumsTester       ( new gctTestHfEtSums() ),
  m_inputEmCands(), m_inputRegions(),
  m_bxStart(0), m_numOfBx(1)
{
}

gctTestFunctions::~gctTestFunctions() {
  delete theElectronsTester;
  delete theEnergyAlgosTester;
  delete theFirmwareTester;
  delete theHtAndJetCountsTester;
  delete theHfEtSumsTester;
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
  theHtAndJetCountsTester->setBxRange(m_bxStart, m_numOfBx);
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
  theHtAndJetCountsTester->fillRawJetData(gct);
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
  return theHtAndJetCountsTester->checkHtSums(gct);
}

//=================================================================================================================
//
/// Check the jet counting algorithms
bool gctTestFunctions::checkJetCounts(const L1GlobalCaloTrigger* gct) const
{
  return theHtAndJetCountsTester->checkJetCounts(gct);
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
