#ifndef DataRecord_L1GtPsbSetupRcd_h
#define DataRecord_L1GtPsbSetupRcd_h

/**
 * \class L1GtPsbSetupRcd
 *
 *
 * Description: record for the setup of L1 GT PSB boards.
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
// class L1GtPsbSetupRcd :
//    public edm::eventsetup::EventSetupRecordImplementation<L1GtPsbSetupRcd>
// {

//     // empty

// };
class L1GtPsbSetupRcd
    : public edm::eventsetup::DependentRecordImplementation<L1GtPsbSetupRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
