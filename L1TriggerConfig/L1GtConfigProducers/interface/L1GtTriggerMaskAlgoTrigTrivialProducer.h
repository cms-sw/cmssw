#ifndef L1GtConfigProducers_L1GtTriggerMaskAlgoTrigTrivialProducer_h
#define L1GtConfigProducers_L1GtTriggerMaskAlgoTrigTrivialProducer_h

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
class L1GtTriggerMaskAlgoTrigRcd;

// class declaration
class L1GtTriggerMaskAlgoTrigTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtTriggerMaskAlgoTrigTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMaskAlgoTrigTrivialProducer() override;


    /// public methods

    std::unique_ptr<L1GtTriggerMask> produceTriggerMask(
        const L1GtTriggerMaskAlgoTrigRcd&);

private:

    /// trigger mask
    std::vector<unsigned int> m_triggerMask;

};

#endif
