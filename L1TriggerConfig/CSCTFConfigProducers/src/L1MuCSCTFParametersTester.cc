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
 * $Date: 2010/05/21 13:04:49 $
 * $Revision: 1.1 $
 *
 */

// this class header
#include "L1TriggerConfig/CSCTFConfigProducers/interface/L1MuCSCTFParametersTester.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

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
L1MuCSCTFParametersTester::L1MuCSCTFParametersTester(const edm::ParameterSet& parSet)
{
    // empty
}

// destructor
L1MuCSCTFParametersTester::~L1MuCSCTFParametersTester()
{
    // empty
}

// loop over events
void L1MuCSCTFParametersTester::analyze(
    const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    edm::ESHandle< L1MuCSCTFConfiguration > l1CSCTFPar ;
    evSetup.get< L1MuCSCTFConfigurationRcd >().get( l1CSCTFPar ) ;

    l1CSCTFPar->print(std::cout);
}
