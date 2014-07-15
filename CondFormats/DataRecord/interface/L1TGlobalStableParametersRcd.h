#ifndef DataRecord_L1TGlobalStableParametersRcd_h
#define DataRecord_L1TGlobalStableParametersRcd_h

/**
 * \class L1TGlobalStableParametersRcd
 * 
 * 
 * Description: record for L1 GT stable parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date: 2007/09/27 10:30:00 $
 * $Revision: 1.1 $
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
// class L1TGlobalStableParametersRcd
//             : public edm::eventsetup::EventSetupRecordImplementation<L1TGlobalStableParametersRcd>
// {

//     // empty

// };
class L1TGlobalStableParametersRcd : public edm::eventsetup::DependentRecordImplementation<L1TGlobalStableParametersRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
