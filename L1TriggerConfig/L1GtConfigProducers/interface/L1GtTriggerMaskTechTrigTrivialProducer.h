#ifndef L1GtConfigProducers_L1GtTriggerMaskTechTrigTrivialProducer_h
#define L1GtConfigProducers_L1GtTriggerMaskTechTrigTrivialProducer_h

/**
 * \class L1GtTriggerMaskTechTrigTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT trigger mask for technical triggers.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

// forward declarations
class L1GtTriggerMaskTechTrigRcd;

// class declaration
class L1GtTriggerMaskTechTrigTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtTriggerMaskTechTrigTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMaskTechTrigTrivialProducer() override;


    /// public methods

    std::unique_ptr<L1GtTriggerMask> produceTriggerMask(
        const L1GtTriggerMaskTechTrigRcd&);

private:

    /// trigger mask
    std::vector<unsigned int> m_triggerMask;

};

#endif
