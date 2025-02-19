#ifndef DataRecord_L1GtTriggerMaskTechTrigRcd_h
#define DataRecord_L1GtTriggerMaskTechTrigRcd_h

/**
 * \class L1GtTriggerMaskTechTrigRcd
 * 
 * 
 * Description: record for L1 GT mask for technical triggers.  
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
// class L1GtTriggerMaskTechTrigRcd
//             : public edm::eventsetup::EventSetupRecordImplementation<L1GtTriggerMaskTechTrigRcd>
// {

//     // empty

// };
class L1GtTriggerMaskTechTrigRcd : public edm::eventsetup::DependentRecordImplementation<L1GtTriggerMaskTechTrigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
