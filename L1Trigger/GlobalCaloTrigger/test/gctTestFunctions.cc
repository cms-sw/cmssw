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

gctTestFunctions::gctTestFunctions() {
  theElectronsTester      = new gctTestElectrons();
  theEnergyAlgosTester    = new gctTestEnergyAlgos();
  theFirmwareTester       = new gctTestFirmware();
  theHtAndJetCountsTester = new gctTestHtAndJetCounts();
  theHfEtSumsTester       = new gctTestHfEtSums();
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
}

//=================================================================================================================
//
/// Load another event into the gct. Overloaded for the various ways of doing this.
void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const bool simpleEvent)
{
  m_inputRegions = theEnergyAlgosTester->loadEvent(gct, simpleEvent);
}

void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const std::string fileName, bool &endOfFile)
{
  m_inputRegions = theEnergyAlgosTester->loadEvent(gct, fileName, endOfFile);
}

void gctTestFunctions::loadNextEvent(L1GlobalCaloTrigger* &gct, const std::string fileName)
{
  m_inputEmCands = theElectronsTester->loadEvent(gct, fileName);
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
  theFirmwareTester->fillJetsFromFirmware(fileName);
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
  return theElectronsTester->checkElectrons(gct);
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
  return theHfEtSumsTester->checkHfEtSums(gct);
}

