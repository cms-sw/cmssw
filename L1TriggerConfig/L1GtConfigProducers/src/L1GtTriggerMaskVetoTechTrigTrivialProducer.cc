/**
 * \class L1GtTriggerMaskVetoTechTrigTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT trigger veto mask for technical triggers.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMaskVetoTechTrigTrivialProducer.h"

// system include files
#include <memory>

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

// forward declarations

// constructor(s)
L1GtTriggerMaskVetoTechTrigTrivialProducer::L1GtTriggerMaskVetoTechTrigTrivialProducer(
        const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this,
            &L1GtTriggerMaskVetoTechTrigTrivialProducer::produceTriggerMask);

    // now do what ever other initialization is needed

    m_triggerMask =
            parSet.getParameter<std::vector<unsigned int> >("TriggerMask");

}

// destructor
L1GtTriggerMaskVetoTechTrigTrivialProducer::~L1GtTriggerMaskVetoTechTrigTrivialProducer()
{

    // empty

}

// member functions

// method called to produce the data
std::unique_ptr<L1GtTriggerMask> L1GtTriggerMaskVetoTechTrigTrivialProducer::produceTriggerMask(
        const L1GtTriggerMaskVetoTechTrigRcd& iRecord)
{
    return std::make_unique<L1GtTriggerMask>(m_triggerMask);
}
