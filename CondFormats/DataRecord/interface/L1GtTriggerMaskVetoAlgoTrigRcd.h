#ifndef DataRecord_L1GtTriggerMaskVetoAlgoTrigRcd_h
#define DataRecord_L1GtTriggerMaskVetoAlgoTrigRcd_h

/**
 * \class L1GtTriggerMaskVetoAlgoTrigRcd
 * 
 * 
 * Description: record for L1 GT veto mask for algorithm triggers.  
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
// class L1GtTriggerMaskVetoAlgoTrigRcd
//             : public edm::eventsetup::EventSetupRecordImplementation<L1GtTriggerMaskVetoAlgoTrigRcd>
// {

//     // empty

// };
class L1GtTriggerMaskVetoAlgoTrigRcd : public edm::eventsetup::DependentRecordImplementation<L1GtTriggerMaskVetoAlgoTrigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
