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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTrivialProducer.h"

// system include files
#include <memory>

#include "boost/shared_ptr.hpp"
#include <boost/cstdint.hpp>


// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

// forward declarations

// constructor(s)
L1GtParametersTrivialProducer::L1GtParametersTrivialProducer(const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &L1GtParametersTrivialProducer::produceGtParameters);

    // now do what ever other initialization is needed

    // total Bx's in the event

    m_totalBxInEvent = parSet.getParameter<int>("TotalBxInEvent");

    if (m_totalBxInEvent > 0) {
        if ( (m_totalBxInEvent%2) == 0 ) {
            m_totalBxInEvent = m_totalBxInEvent - 1;

            edm::LogInfo("L1GtParametersTrivialProducer")
            << "\nWARNING: Number of bunch crossing in event rounded to: "
            << m_totalBxInEvent << "\n         The number must be an odd number!\n"
            << std::endl;
        }
    } else {

        edm::LogInfo("L1GtParametersTrivialProducer")
        << "\nWARNING: Number of bunch crossing in event must be a positive number!"
        << "\n  Requested value was: " << m_totalBxInEvent
        << "\n  Reset to 1 (L1Accept bunch only).\n"
        << std::endl;

        m_totalBxInEvent = 1;

    }

    m_activeBoards =
        static_cast<boost::uint16_t>(parSet.getParameter<unsigned int>("ActiveBoards"));

}

// destructor
L1GtParametersTrivialProducer::~L1GtParametersTrivialProducer()
{

    // empty

}


// member functions

// method called to produce the data
boost::shared_ptr<L1GtParameters> L1GtParametersTrivialProducer::produceGtParameters(
    const L1GtParametersRcd& iRecord)
{

    using namespace edm::es;


    boost::shared_ptr<L1GtParameters> pL1GtParameters =
        boost::shared_ptr<L1GtParameters>( new L1GtParameters() );


    // set total Bx's in the event
    pL1GtParameters->setGtTotalBxInEvent(m_totalBxInEvent);

    // set the active boards
    pL1GtParameters->setGtActiveBoards(m_activeBoards);


    return pL1GtParameters ;
}

