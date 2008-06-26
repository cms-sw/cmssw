#ifndef L1GtConfigProducers_L1GtPrescaleFactorsTechTrigTrivialProducer_h
#define L1GtConfigProducers_L1GtPrescaleFactorsTechTrigTrivialProducer_h

/**
 * \class L1GtPrescaleFactorsTechTrigTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT prescale factors for technical triggers.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
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

// forward declarations
class L1GtPrescaleFactorsTechTrigRcd;

// class declaration
class L1GtPrescaleFactorsTechTrigTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtPrescaleFactorsTechTrigTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtPrescaleFactorsTechTrigTrivialProducer();

    /// public methods

    boost::shared_ptr<L1GtPrescaleFactors> producePrescaleFactors(
            const L1GtPrescaleFactorsTechTrigRcd&);

private:

    /// prescale factors
    std::vector<std::vector<int> > m_prescaleFactors;

};

#endif
