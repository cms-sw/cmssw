#ifndef DataRecord_L1GtPrescaleFactorsRcd_h
#define DataRecord_L1GtPrescaleFactorsRcd_h

/**
 * \class L1GtPrescaleFactorsRcd
 * 
 * 
 * Description: record for L1 GT prescale factors.  
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
// class L1GtPrescaleFactorsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GtPrescaleFactorsRcd>
// {

//     // empty

// };
class L1GtPrescaleFactorsRcd : public edm::eventsetup::DependentRecordImplementation<L1GtPrescaleFactorsRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
