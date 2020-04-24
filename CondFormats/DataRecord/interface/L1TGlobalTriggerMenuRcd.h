#ifndef DataRecord_L1TGlobalTriggerMenuRcd_h
#define DataRecord_L1TGlobalTriggerMenuRcd_h

/**
 * \class L1TGlobalTriggerMenuRcd
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
#include "CondFormats/DataRecord/interface/L1TGlobalStableParametersRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations

// class declaration - record depends on L1TGlobalStableParametersRcd
class L1TGlobalTriggerMenuRcd : public edm::eventsetup::DependentRecordImplementation<
            L1TGlobalTriggerMenuRcd, boost::mpl::vector<L1TGlobalStableParametersRcd,L1TriggerKeyListRcd,L1TriggerKeyRcd> >
{

    // empty

};

#endif
