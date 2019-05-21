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
L1GtPsbSetupTester::L1GtPsbSetupTester(const edm::ParameterSet& parSet) {
  // empty
}

// destructor
L1GtPsbSetupTester::~L1GtPsbSetupTester() {
  // empty
}

// loop over events
void L1GtPsbSetupTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1GtPsbSetup> l1GtPsbSet;
  evSetup.get<L1GtPsbSetupRcd>().get(l1GtPsbSet);

  l1GtPsbSet->print(std::cout);
  std::cout << std::endl;
}
