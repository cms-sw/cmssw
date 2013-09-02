/**
 * \class L1GtTriggerMaskAlgoTrigTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT trigger mask for algorithm triggers.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMaskAlgoTrigTrivialProducer.h"

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"

// forward declarations

// constructor(s)
L1GtTriggerMaskAlgoTrigTrivialProducer::L1GtTriggerMaskAlgoTrigTrivialProducer(
        const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this,
            &L1GtTriggerMaskAlgoTrigTrivialProducer::produceTriggerMask);

    // now do what ever other initialization is needed

    m_triggerMask =
            parSet.getParameter<std::vector<unsigned int> >("TriggerMask");

}

// destructor
L1GtTriggerMaskAlgoTrigTrivialProducer::~L1GtTriggerMaskAlgoTrigTrivialProducer()
{

    // empty

}

// member functions

// method called to produce the data
boost::shared_ptr<L1GtTriggerMask> L1GtTriggerMaskAlgoTrigTrivialProducer::produceTriggerMask(
        const L1GtTriggerMaskAlgoTrigRcd& iRecord)
{
    boost::shared_ptr<L1GtTriggerMask> pL1GtTriggerMask = boost::shared_ptr<L1GtTriggerMask>(
            new L1GtTriggerMask(m_triggerMask) );

    return pL1GtTriggerMask ;
}
