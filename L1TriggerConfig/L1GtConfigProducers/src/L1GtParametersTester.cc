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
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

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
L1GtParametersTester::L1GtParametersTester(const edm::ParameterSet& parSet) {
  // empty
}

// destructor
L1GtParametersTester::~L1GtParametersTester() {
  // empty
}

// loop over events
void L1GtParametersTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1GtParameters> l1GtPar;
  evSetup.get<L1GtParametersRcd>().get(l1GtPar);

  LogDebug("L1GtParametersTester") << (*l1GtPar) << std::endl;
}
