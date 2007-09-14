/**
 * \class L1GtFactorsTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT prescale factors and masks.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTrivialProducer.h"

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include <vector>


// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskRcd.h"

// forward declarations

// constructor(s)
L1GtFactorsTrivialProducer::L1GtFactorsTrivialProducer(const edm::ParameterSet& iConfig)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &L1GtFactorsTrivialProducer::producePrescaleFactors);
    setWhatProduced(this, &L1GtFactorsTrivialProducer::produceTriggerMask);

    // now do what ever other initialization is needed

    // prescale factors
    m_prescaleFactors =
        iConfig.getParameter<std::vector<int> >("PrescaleFactors");

    // trigger mask
    m_triggerMask =
        iConfig.getParameter<std::vector<unsigned int> >("TriggerMask");


}

// destructor
L1GtFactorsTrivialProducer::~L1GtFactorsTrivialProducer()
{

    // empty

}


// member functions

// method called to produce the data
boost::shared_ptr<L1GtPrescaleFactors> L1GtFactorsTrivialProducer::producePrescaleFactors(
    const L1GtPrescaleFactorsRcd& iRecord)
{

    using namespace edm::es;

    boost::shared_ptr<L1GtPrescaleFactors> pL1GtPrescaleFactors =
        boost::shared_ptr<L1GtPrescaleFactors>(
            new L1GtPrescaleFactors(m_prescaleFactors) );

    return pL1GtPrescaleFactors ;
}

// method called to produce the data
boost::shared_ptr<L1GtTriggerMask> L1GtFactorsTrivialProducer::produceTriggerMask(
    const L1GtTriggerMaskRcd& iRecord)
{

    using namespace edm::es;

    boost::shared_ptr<L1GtTriggerMask> pL1GtTriggerMask =
        boost::shared_ptr<L1GtTriggerMask>(
            new L1GtTriggerMask(m_triggerMask) );

    return pL1GtTriggerMask ;
}
