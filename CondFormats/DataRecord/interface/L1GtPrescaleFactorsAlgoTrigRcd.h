#ifndef DataRecord_L1GtPrescaleFactorsAlgoTrigRcd_h
#define DataRecord_L1GtPrescaleFactorsAlgoTrigRcd_h

/**
 * \class L1GtPrescaleFactorsAlgoTrigRcd
 * 
 * 
 * Description: record for L1 GT prescale factors for algorithm triggers.  
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
// class L1GtPrescaleFactorsAlgoTrigRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GtPrescaleFactorsAlgoTrigRcd>
// {

//     // empty

// };
class L1GtPrescaleFactorsAlgoTrigRcd
    : public edm::eventsetup::DependentRecordImplementation<L1GtPrescaleFactorsAlgoTrigRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
