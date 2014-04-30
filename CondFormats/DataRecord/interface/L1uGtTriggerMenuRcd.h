#ifndef DataRecord_L1uGtTriggerMenuRcd_h
#define DataRecord_L1uGtTriggerMenuRcd_h

/**
 * \class L1uGtTriggerMenuRcd
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
#include "boost/mpl/vector.hpp"

// user include files
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1uGtStableParametersRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations

// class declaration - record depends on L1uGtStableParametersRcd
class L1uGtTriggerMenuRcd : public edm::eventsetup::DependentRecordImplementation<
            L1uGtTriggerMenuRcd, boost::mpl::vector<L1uGtStableParametersRcd,L1TriggerKeyListRcd,L1TriggerKeyRcd> >
{

    // empty

};

#endif
