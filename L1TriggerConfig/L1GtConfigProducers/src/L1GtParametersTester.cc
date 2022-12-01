/**
 * \class L1GtParametersTester
 *
 *
 * Description: test analyzer for L1 GT parameters.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTester.h"

// system include files
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// forward declarations

// constructor(s)
L1GtParametersTester::L1GtParametersTester(const edm::ParameterSet& parSet) : m_l1GtParToken(esConsumes()) {
  // empty
}

// loop over events
void L1GtParametersTester::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& evSetup) const {
  LogDebug("L1GtParametersTester") << evSetup.getData(m_l1GtParToken) << std::endl;
}
