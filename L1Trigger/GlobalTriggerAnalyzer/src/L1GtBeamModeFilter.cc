/**
 * \class L1GtBeamModeFilter
 *
 *
 * Description: see class header.
 *
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtBeamModeFilter.h"

// system include files
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include <cstdint>

// constructor(s)
L1GtBeamModeFilter::L1GtBeamModeFilter(const edm::ParameterSet& parSet)
    :

      m_condInEdmInputTag(parSet.getParameter<edm::InputTag>("CondInEdmInputTag")),
      m_l1GtEvmReadoutRecordTag(parSet.getParameter<edm::InputTag>("L1GtEvmReadoutRecordTag")),
      m_allowedBeamMode(parSet.getParameter<std::vector<unsigned int> >("AllowedBeamMode")),
      m_invertResult(parSet.getParameter<bool>("InvertResult")),
      m_isDebugEnabled(edm::isDebugEnabled()),
      m_condInRunBlockValid(false) {
  if (m_isDebugEnabled) {
    LogDebug("L1GtBeamModeFilter") << std::endl;

    LogTrace("L1GtBeamModeFilter") << "\nInput tag for ConditionsInEdm product: " << m_condInEdmInputTag
                                   << "\nInput tag for L1 GT EVM record:        " << m_l1GtEvmReadoutRecordTag
                                   << "\nAllowed beam modes:" << std::endl;

    for (std::vector<unsigned int>::const_iterator itMode = m_allowedBeamMode.begin();
         itMode != m_allowedBeamMode.end();
         ++itMode) {
      LogTrace("L1GtBeamModeFilter") << "  " << (*itMode) << std::endl;
    }

    LogTrace("L1GtBeamModeFilter") << "\nInvert result (use as NOT filter): " << m_invertResult << std::endl;

    LogTrace("L1GtBeamModeFilter") << std::endl;
  }
}

// destructor
L1GtBeamModeFilter::~L1GtBeamModeFilter() {
  // empty now
}

// member functions

bool L1GtBeamModeFilter::filter(edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // initialize filter result
  bool filterResult = false;

  // for MC samples, return always true (not even checking validity of L1GlobalTriggerEvmReadoutRecord)
  // eventually, the BST information will be filled also in MC simulation to spare this check

  if (!(iEvent.isRealData())) {
    //LogDebug("L1GtBeamModeFilter") << "\nRunning on MC sample."
    //        << "\nAll events are automatically selected, without testing the beam mode"
    //        << "\n" << std::endl;

    if (m_invertResult) {
      return false;
    } else {
      return true;
    }
  }

  // data

  // initialize beam mode value
  uint16_t beamModeValue = 0;

  // get Run Data - the same code can be run in beginRun, with getByLabel from edm::Run
  // TODO should I cache the run/luminosity section? to be decided after moving beamMode to LumiBlock
  const edm::Run& iRun = iEvent.getRun();

  // get ConditionsInRunBlock
  edm::Handle<edm::ConditionsInRunBlock> condInRunBlock;
  iRun.getByLabel(m_condInEdmInputTag, condInRunBlock);

  if (!condInRunBlock.isValid()) {
    LogDebug("L1GtBeamModeFilter") << "\nConditionsInRunBlock with \n  " << m_condInEdmInputTag
                                   << "\nrequested in configuration, but not found in the event."
                                   << "\n"
                                   << std::endl;

    m_condInRunBlockValid = false;

  } else {
    m_condInRunBlockValid = true;
    beamModeValue = condInRunBlock->beamMode;
  }

  // fall through to L1GlobalTriggerEvmReadoutRecord if ConditionsInRunBlock
  // not available
  if (!m_condInRunBlockValid) {
    // get L1GlobalTriggerEvmReadoutRecord and beam mode
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
    iEvent.getByLabel(m_l1GtEvmReadoutRecordTag, gtEvmReadoutRecord);

    if (!gtEvmReadoutRecord.isValid()) {
      LogDebug("L1GtBeamModeFilter") << "\nL1GlobalTriggerEvmReadoutRecord with input tag " << m_l1GtEvmReadoutRecordTag
                                     << "\nrequested in configuration, but not found in the event." << std::endl;

      // both products are missing, return false for both normal and inverted result
      return false;

    } else {
      beamModeValue = (gtEvmReadoutRecord->gtfeWord()).beamMode();
    }
  }

  LogDebug("L1GtBeamModeFilter") << "\nBeam mode: " << beamModeValue << std::endl;

  for (std::vector<unsigned int>::const_iterator itMode = m_allowedBeamMode.begin(); itMode != m_allowedBeamMode.end();
       ++itMode) {
    if (beamModeValue == (*itMode)) {
      filterResult = true;

      //LogTrace("L1GtBeamModeFilter") << "Event selected - beam mode: "
      //        << beamModeValue << "\n" << std::endl;

      break;
    }
  }

  //

  if (m_invertResult) {
    filterResult = !filterResult;
  }

  return filterResult;
}
