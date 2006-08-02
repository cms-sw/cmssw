/**
 * \class L1GlobalTriggerConditions
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder, H. Rohringer - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

// system include files
#include <string>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1TriggerObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GlobalTriggerConditions::L1GlobalTriggerConditions(const std::string& name) {

    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ << std::endl; 
    
     p_name = name; 
     p_lastresult = false;

}

// copy constructor
L1GlobalTriggerConditions::L1GlobalTriggerConditions(L1GlobalTriggerConditions& cp) {

    p_name = cp.getName(); 
    
    
} 

// destructor
L1GlobalTriggerConditions::~L1GlobalTriggerConditions() {
}

