/**
 * \class L1GtTriggerMenuTester
 * 
 * 
 * Description: test analyzer for L1 GT trigger menu.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations

// constructor(s)
L1GtTriggerMenuTester::L1GtTriggerMenuTester(const edm::ParameterSet& parSet)
{
    // empty
}

// destructor
L1GtTriggerMenuTester::~L1GtTriggerMenuTester()
{
    // empty
}

// loop over events
void L1GtTriggerMenuTester::analyze(
    const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    edm::ESHandle< L1GtTriggerMenu > l1GtMenu ;
    evSetup.get< L1GtTriggerMenuRcd >().get( l1GtMenu ) ;

    // print with various level of verbosities

    int printVerbosity = 0;
    l1GtMenu->print(std::cout, printVerbosity);

    printVerbosity = 1;
    l1GtMenu->print(std::cout, printVerbosity);

    printVerbosity = 2;
    l1GtMenu->print(std::cout, printVerbosity);

}
