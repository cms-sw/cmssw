/**
 * \class L1GtPsbSetupTester
 *
 *
 * Description: test analyzer for the setup of L1 GT PSB boards.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPsbSetupTester.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files
//   base class

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"
#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"

// forward declarations

// constructor(s)
L1GtPsbSetupTester::L1GtPsbSetupTester(const edm::ParameterSet& parSet) : m_getToken(esConsumes()) {
  // empty
}

// loop over events
void L1GtPsbSetupTester::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& evSetup) const {
  evSetup.getData(m_getToken).print(std::cout);
  std::cout << std::endl;
}
