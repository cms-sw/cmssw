/**
 * \class L1GtFactorsTester
 * 
 * 
 * Description: test analyzer for L1 GT prescale factors and masks.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTester.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskRcd.h"

// forward declarations

// constructor(s)
L1GtFactorsTester::L1GtFactorsTester(const edm::ParameterSet& parSet)
{
    // empty
}

// destructor
L1GtFactorsTester::~L1GtFactorsTester()
{
    // empty
}

// loop over events
void L1GtFactorsTester::analyze(
    const edm::Event& iEvent, const edm::EventSetup& evSetup)
{


    edm::ESHandle< L1GtPrescaleFactors > l1GtPF ;
    evSetup.get< L1GtPrescaleFactorsRcd >().get( l1GtPF ) ;

    l1GtPF->print(std::cout);


    edm::ESHandle< L1GtTriggerMask > l1GtTM ;
    evSetup.get< L1GtTriggerMaskRcd >().get( l1GtTM ) ;

    l1GtTM->print(std::cout);

}
