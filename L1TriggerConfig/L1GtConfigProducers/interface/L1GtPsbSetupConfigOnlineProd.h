#ifndef L1GtConfigProducers_L1GtPsbSetupConfigOnlineProd_h
#define L1GtConfigProducers_L1GtPsbSetupConfigOnlineProd_h

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

// system include files
#include "boost/shared_ptr.hpp"
#include <string>

// user include files
//   base class
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"
#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"

// forward declarations

// class declaration
class L1GtPsbSetupConfigOnlineProd :
        public L1ConfigOnlineProdBase<L1GtPsbSetupRcd, L1GtPsbSetup>
{

public:

    /// constructor
    L1GtPsbSetupConfigOnlineProd(const edm::ParameterSet&);

    /// destructor
    ~L1GtPsbSetupConfigOnlineProd();

    /// public methods
    virtual boost::shared_ptr<L1GtPsbSetup> newObject(const std::string& objectKey);

private:

    ///
};

#endif
