#ifndef L1GtConfigProducers_L1GtParametersTrivialProducer_h
#define L1GtConfigProducers_L1GtParametersTrivialProducer_h

/**
 * \class L1GtParametersTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT parameters.  
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
#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

// forward declarations

// class declaration
class L1GtParametersTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtParametersTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtParametersTrivialProducer();


    /// public methods

    /// L1 GT parameters
    boost::shared_ptr<L1GtParameters> produceGtParameters(
        const L1GtParametersRcd&);

private:

    /// total Bx's in the event
    int m_totalBxInEvent;

    /// active boards
    boost::uint16_t m_activeBoards;

};

#endif
