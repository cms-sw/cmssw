/**
 * \class L1GtPrescaleFactorsAndMasksTester
 * 
 * 
 * Description: test analyzer for L1 GT prescale factors and masks.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPrescaleFactorsAndMasksTester.h"

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

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

// forward declarations

// constructor(s)
L1GtPrescaleFactorsAndMasksTester::L1GtPrescaleFactorsAndMasksTester(
        const edm::ParameterSet& parSet) {
    // empty
}

// destructor
L1GtPrescaleFactorsAndMasksTester::~L1GtPrescaleFactorsAndMasksTester() {
    // empty
}

// loop over events
void L1GtPrescaleFactorsAndMasksTester::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    //
    edm::ESHandle< L1GtPrescaleFactors> l1GtPfAlgo;
    evSetup.get< L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo) ;

    std::cout << "\nL1 GT prescale factors for algorithm triggers" << std::endl;
    l1GtPfAlgo->print(std::cout);

    edm::ESHandle< L1GtPrescaleFactors> l1GtPfTech;
    evSetup.get< L1GtPrescaleFactorsTechTrigRcd>().get(l1GtPfTech) ;

    std::cout << "\nL1 GT prescale factors for technical triggers" << std::endl;
    l1GtPfTech->print(std::cout);

    // 
    edm::ESHandle< L1GtTriggerMask> l1GtTmAlgo;
    evSetup.get< L1GtTriggerMaskAlgoTrigRcd>().get(l1GtTmAlgo) ;

    std::cout << "\nL1 GT trigger masks for algorithm triggers" << std::endl;
    l1GtTmAlgo->print(std::cout);

    edm::ESHandle< L1GtTriggerMask> l1GtTmTech;
    evSetup.get< L1GtTriggerMaskTechTrigRcd>().get(l1GtTmTech) ;

    std::cout << "\nL1 GT trigger masks for technical triggers" << std::endl;
    l1GtTmTech->print(std::cout);

    // 
    edm::ESHandle< L1GtTriggerMask> l1GtTmVetoAlgo;
    evSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd>().get(l1GtTmVetoAlgo) ;

    std::cout << "\nL1 GT trigger veto masks for algorithm triggers"
            << std::endl;
    l1GtTmVetoAlgo->print(std::cout);

    edm::ESHandle< L1GtTriggerMask> l1GtTmVetoTech;
    evSetup.get< L1GtTriggerMaskVetoTechTrigRcd>().get(l1GtTmVetoTech) ;

    std::cout << "\nL1 GT trigger veto masks for technical triggers"
            << std::endl;
    l1GtTmVetoTech->print(std::cout);

}
