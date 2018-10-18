#ifndef L1GtConfigProducers_L1GtTriggerMaskVetoAlgoTrigTrivialProducer_h
#define L1GtConfigProducers_L1GtTriggerMaskVetoAlgoTrigTrivialProducer_h

/**
 * \class L1GtTriggerMaskVetoAlgoTrigTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT trigger veto mask for algorithm triggers.  
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
class L1GtTriggerMaskVetoAlgoTrigRcd;

// class declaration
class L1GtTriggerMaskVetoAlgoTrigTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtTriggerMaskVetoAlgoTrigTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMaskVetoAlgoTrigTrivialProducer() override;


    /// public methods

    std::unique_ptr<L1GtTriggerMask> produceTriggerMask(
        const L1GtTriggerMaskVetoAlgoTrigRcd&);

private:

    /// trigger mask
    std::vector<unsigned int> m_triggerMask;

};

#endif
