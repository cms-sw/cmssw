#ifndef L1GtConfigProducers_L1GtPrescaleFactorsAlgoTrigConfigOnlineProd_h
#define L1GtConfigProducers_L1GtPrescaleFactorsAlgoTrigConfigOnlineProd_h

/**
 * \class L1GtPrescaleFactorsAlgoTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPrescaleFactorsAlgoTrigRcd.
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

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"

// forward declarations

// class declaration
class L1GtPrescaleFactorsAlgoTrigConfigOnlineProd :
        public L1ConfigOnlineProdBase<L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors>
{

public:

    /// constructor
    L1GtPrescaleFactorsAlgoTrigConfigOnlineProd(const edm::ParameterSet&);

    /// destructor
    ~L1GtPrescaleFactorsAlgoTrigConfigOnlineProd();

    /// public methods
    virtual boost::shared_ptr<L1GtPrescaleFactors> newObject(const std::string& objectKey);

private:

    bool m_isDebugEnabled;

};

#endif
