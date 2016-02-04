#ifndef DataRecord_L1GtTriggerMaskVetoTechTrigRcd_h
#define DataRecord_L1GtTriggerMaskVetoTechTrigRcd_h

/**
 * \class L1GtTriggerMaskVetoTechTrigRcd
 * 
 * 
 * Description: record for L1 GT veto mask for technical triggers.  
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
// class L1GtTriggerMaskVetoTechTrigRcd
//             : public edm::eventsetup::EventSetupRecordImplementation<L1GtTriggerMaskVetoTechTrigRcd>
// {

//     // empty

// };
class L1GtTriggerMaskVetoTechTrigRcd : public edm::eventsetup::DependentRecordImplementation<L1GtTriggerMaskVetoTechTrigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
