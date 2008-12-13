/**
 * \class L1GtPsbSetupConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPsbSetup.
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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPsbSetupConfigOnlineProd.h"

// system include files
#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtPsbSetupConfigOnlineProd::L1GtPsbSetupConfigOnlineProd(const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtPsbSetupRcd, L1GtPsbSetup> (parSet) {

    // empty

}

// destructor
L1GtPsbSetupConfigOnlineProd::~L1GtPsbSetupConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtPsbSetup> L1GtPsbSetupConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtPsbSetup
    boost::shared_ptr<L1GtPsbSetup> pL1GtPsbSetup = boost::shared_ptr<L1GtPsbSetup>(
            new L1GtPsbSetup());

    return pL1GtPsbSetup;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtPsbSetupConfigOnlineProd);
