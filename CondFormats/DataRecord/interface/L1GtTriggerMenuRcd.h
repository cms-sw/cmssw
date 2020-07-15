#ifndef DataRecord_L1GtTriggerMenuRcd_h
#define DataRecord_L1GtTriggerMenuRcd_h

/**
 * \class L1GtTriggerMenuRcd
 * 
 * 
 * Description: record for L1 trigger menu.  
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
#include <boost/mp11/list.hpp>

// user include files
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations

// class declaration - record depends on L1GtStableParametersRcd
class L1GtTriggerMenuRcd : public edm::eventsetup::DependentRecordImplementation<
                               L1GtTriggerMenuRcd,
                               boost::mp11::mp_list<L1GtStableParametersRcd, L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
  // empty
};

#endif
