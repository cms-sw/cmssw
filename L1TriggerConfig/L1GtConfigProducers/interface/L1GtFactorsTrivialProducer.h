#ifndef L1GtConfigProducers_L1GtFactorsTrivialProducer_h
#define L1GtConfigProducers_L1GtFactorsTrivialProducer_h

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

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskRcd.h"

// forward declarations

// class declaration
class L1GtFactorsTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtFactorsTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtFactorsTrivialProducer();


    /// public methods

    /// L1 GT prescale factors
    boost::shared_ptr<L1GtPrescaleFactors> producePrescaleFactors(
        const L1GtPrescaleFactorsRcd&);

    /// L1 GT trigger mask
    boost::shared_ptr<L1GtTriggerMask> produceTriggerMask(
        const L1GtTriggerMaskRcd&);

private:

    /// prescale factors
    std::vector<int> m_prescaleFactors;

    /// trigger mask
    std::vector<unsigned int> m_triggerMask;

};

#endif
