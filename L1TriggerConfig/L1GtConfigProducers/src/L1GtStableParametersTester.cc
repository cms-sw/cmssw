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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

// forward declarations

// constructor(s)
L1GtStableParametersTester::L1GtStableParametersTester(const edm::ParameterSet& parSet) : m_l1GtParToken(esConsumes()) {
  // empty
}

// loop over events
void L1GtStableParametersTester::analyze(edm::StreamID,
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& evSetup) const {
  evSetup.getData(m_l1GtParToken).print(std::cout);
}
