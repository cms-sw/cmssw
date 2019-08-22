/**
 * \class L1GtStableParametersTester
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtStableParametersTester.h"

// system include files
#include <iomanip>
#include <iostream>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

// forward declarations

// constructor(s)
L1GtStableParametersTester::L1GtStableParametersTester(const edm::ParameterSet& parSet) {
  // empty
}

// destructor
L1GtStableParametersTester::~L1GtStableParametersTester() {
  // empty
}

// loop over events
void L1GtStableParametersTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1GtStableParameters> l1GtPar;
  evSetup.get<L1GtStableParametersRcd>().get(l1GtPar);

  l1GtPar->print(std::cout);
}
