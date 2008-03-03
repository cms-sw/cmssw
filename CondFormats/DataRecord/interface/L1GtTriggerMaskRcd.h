#ifndef DataRecord_L1GtTriggerMaskRcd_h
#define DataRecord_L1GtTriggerMaskRcd_h

/**
 * \class L1GtTriggerMaskRcd
 * 
 * 
 * Description: record for L1 GT mask.  
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
// class L1GtTriggerMaskRcd
//             : public edm::eventsetup::EventSetupRecordImplementation<L1GtTriggerMaskRcd>
// {

//     // empty

// };
class L1GtTriggerMaskRcd : public edm::eventsetup::DependentRecordImplementation<L1GtTriggerMaskRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
