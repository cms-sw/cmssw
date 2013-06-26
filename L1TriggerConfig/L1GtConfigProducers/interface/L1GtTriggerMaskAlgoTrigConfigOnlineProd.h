#ifndef L1GtConfigProducers_L1GtTriggerMaskAlgoTrigConfigOnlineProd_h
#define L1GtConfigProducers_L1GtTriggerMaskAlgoTrigConfigOnlineProd_h

/**
 * \class L1GtTriggerMaskAlgoTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMaskAlgoTrigRcd.
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

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"

// forward declarations

// class declaration
class L1GtTriggerMaskAlgoTrigConfigOnlineProd :
        public L1ConfigOnlineProdBase<L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask>
{

public:

    /// constructor
    L1GtTriggerMaskAlgoTrigConfigOnlineProd(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMaskAlgoTrigConfigOnlineProd();

    /// public methods
    virtual boost::shared_ptr<L1GtTriggerMask> newObject(const std::string& objectKey);

private:

    /// partition number
    int m_partitionNumber;

};

#endif
