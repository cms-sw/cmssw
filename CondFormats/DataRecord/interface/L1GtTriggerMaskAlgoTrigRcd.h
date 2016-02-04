#ifndef DataRecord_L1GtTriggerMaskAlgoTrigRcd_h
#define DataRecord_L1GtTriggerMaskAlgoTrigRcd_h

/**
 * \class L1GtTriggerMaskAlgoTrigRcd
 * 
 * 
 * Description: record for L1 GT mask for algorithm triggers.  
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
// class L1GtTriggerMaskAlgoTrigRcd
//             : public edm::eventsetup::EventSetupRecordImplementation<L1GtTriggerMaskAlgoTrigRcd>
// {

//     // empty

// };
class L1GtTriggerMaskAlgoTrigRcd : public edm::eventsetup::DependentRecordImplementation<L1GtTriggerMaskAlgoTrigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
