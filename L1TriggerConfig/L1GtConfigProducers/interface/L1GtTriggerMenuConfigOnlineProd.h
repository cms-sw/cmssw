#ifndef L1GtConfigProducers_L1GtTriggerMenuConfigOnlineProd_h
#define L1GtConfigProducers_L1GtTriggerMenuConfigOnlineProd_h

/**
 * \class L1GtTriggerMenuConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMenu.
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

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations

// class declaration
class L1GtTriggerMenuConfigOnlineProd :
        public L1ConfigOnlineProdBase<L1GtTriggerMenuRcd, L1GtTriggerMenu>
{

public:

    /// constructor
    L1GtTriggerMenuConfigOnlineProd(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMenuConfigOnlineProd();

    /// public methods
    virtual boost::shared_ptr<L1GtTriggerMenu> newObject(const std::string& objectKey);

private:

    ///
};

#endif
