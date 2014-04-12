#ifndef DataRecord_L1GtParametersRcd_h
#define DataRecord_L1GtParametersRcd_h

/**
 * \class L1GtParametersRcd
 * 
 * 
 * Description: record for L1 GT parameters.  
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
#include "boost/mpl/vector.hpp"

// user include files
//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations

// class declaration
// class L1GtParametersRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GtParametersRcd>
// {

//     // empty

// };
class L1GtParametersRcd : public edm::eventsetup::DependentRecordImplementation<L1GtParametersRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
