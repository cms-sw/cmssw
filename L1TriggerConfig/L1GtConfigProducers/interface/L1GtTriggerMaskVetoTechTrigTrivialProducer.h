#ifndef L1GtConfigProducers_L1GtTriggerMaskVetoTechTrigTrivialProducer_h
#define L1GtConfigProducers_L1GtTriggerMaskVetoTechTrigTrivialProducer_h

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
class L1GtTriggerMaskVetoTechTrigRcd;

// class declaration
class L1GtTriggerMaskVetoTechTrigTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtTriggerMaskVetoTechTrigTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMaskVetoTechTrigTrivialProducer() override;


    /// public methods

    std::unique_ptr<L1GtTriggerMask> produceTriggerMask(
        const L1GtTriggerMaskVetoTechTrigRcd&);

private:

    /// trigger mask
    std::vector<unsigned int> m_triggerMask;

};

#endif
