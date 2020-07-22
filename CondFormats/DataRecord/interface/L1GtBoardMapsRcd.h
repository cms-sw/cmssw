#ifndef DataRecord_L1GtBoardMapsRcd_h
#define DataRecord_L1GtBoardMapsRcd_h

/**
 * \class L1GtBoardMapsRcd
 * 
 * 
 * Description: record for various mappings of the L1 GT boards.  
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
#include "FWCore/Utilities/interface/mplVector.h"

// user include files
//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

// forward declarations

// class declaration
// class L1GtBoardMapsRcd :
//    public edm::eventsetup::EventSetupRecordImplementation<L1GtBoardMapsRcd>
// {

//     // empty

// };
class L1GtBoardMapsRcd
    : public edm::eventsetup::DependentRecordImplementation<L1GtBoardMapsRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
