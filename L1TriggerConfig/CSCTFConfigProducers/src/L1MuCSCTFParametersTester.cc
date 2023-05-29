/**
 * \class L1MuCSCTFParametersTester
 *
 *
 * Description: test analyzer for L1 CSCTF parameters.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: G.P. Di Giovanni - University of Florida
 *
 *
 */

// this class header
#include "L1TriggerConfig/CSCTFConfigProducers/interface/L1MuCSCTFParametersTester.h"

// system include files
#include <iomanip>
#include <iostream>

// user include files
//   base class

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// forward declarations

// constructor(s)
L1MuCSCTFParametersTester::L1MuCSCTFParametersTester(const edm::ParameterSet& parSet) { token_ = esConsumes(); }

// loop over events
void L1MuCSCTFParametersTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  evSetup.getData(token_).print(std::cout);
}
