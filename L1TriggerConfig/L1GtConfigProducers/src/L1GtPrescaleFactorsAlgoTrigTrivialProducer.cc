/**
 * \class L1GtPrescaleFactorsAlgoTrigTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT prescale factors for algorithm triggers.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPrescaleFactorsAlgoTrigTrivialProducer.h"

// system include files
#include <memory>

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"

// forward declarations

// constructor(s)
L1GtPrescaleFactorsAlgoTrigTrivialProducer::L1GtPrescaleFactorsAlgoTrigTrivialProducer(
        const edm::ParameterSet& parSet) {

    // tell the framework what data is being produced
    setWhatProduced(this,
            &L1GtPrescaleFactorsAlgoTrigTrivialProducer::producePrescaleFactors);

    // now do what ever other initialization is needed

    // prescale factors

    std::vector<edm::ParameterSet> prescaleFactorsSet =
        parSet.getParameter<std::vector<edm::ParameterSet> >("PrescaleFactorsSet");

    for (std::vector<edm::ParameterSet>::const_iterator itPfSet =
            prescaleFactorsSet.begin(); itPfSet != prescaleFactorsSet.end(); ++itPfSet) {

        // prescale factors
        m_prescaleFactors.push_back(itPfSet->getParameter<std::vector<int> >("PrescaleFactors"));

    }

}

// destructor
L1GtPrescaleFactorsAlgoTrigTrivialProducer::~L1GtPrescaleFactorsAlgoTrigTrivialProducer() {

    // empty

}

// member functions

// method called to produce the data
std::unique_ptr<L1GtPrescaleFactors> L1GtPrescaleFactorsAlgoTrigTrivialProducer::producePrescaleFactors(
        const L1GtPrescaleFactorsAlgoTrigRcd& iRecord) {

    return std::make_unique<L1GtPrescaleFactors>(m_prescaleFactors);
}
