#ifndef L1GtConfigProducers_L1GtPrescaleFactorsAlgoTrigTrivialProducer_h
#define L1GtConfigProducers_L1GtPrescaleFactorsAlgoTrigTrivialProducer_h

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

// system include files
#include <memory>

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

// forward declarations
class L1GtPrescaleFactorsAlgoTrigRcd;

// class declaration
class L1GtPrescaleFactorsAlgoTrigTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtPrescaleFactorsAlgoTrigTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtPrescaleFactorsAlgoTrigTrivialProducer() override;

    /// public methods

    std::unique_ptr<L1GtPrescaleFactors> producePrescaleFactors(
            const L1GtPrescaleFactorsAlgoTrigRcd&);

private:

    /// prescale factors
    std::vector<std::vector<int> > m_prescaleFactors;

};

#endif
