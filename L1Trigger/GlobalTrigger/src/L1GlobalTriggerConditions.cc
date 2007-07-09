/**
 * \class L1GlobalTriggerConditions
 * 
 * 
 * Description: see header file.
 *  
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder, H. Rohringer - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

// system include files
#include <string>
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GlobalTriggerConditions::L1GlobalTriggerConditions(
    const L1GlobalTrigger& gt, const std::string& name) 
    :m_GT(gt) 
    {

//    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ 
//        << " condition name: " << name 
//        << std::endl; 
    
     p_name = name; 
     p_lastresult = false;
     p_combinationsInCond = new CombinationsInCond;
     p_objectsInCond = new ObjectTypeInCond;

}

// copy constructor
L1GlobalTriggerConditions::L1GlobalTriggerConditions(L1GlobalTriggerConditions& cp) 
    : m_GT(cp.m_GT)
    {

    p_name = cp.getName(); 
        
} 

// destructor
L1GlobalTriggerConditions::~L1GlobalTriggerConditions() {

    delete p_combinationsInCond;
    delete p_objectsInCond;

}

// methods

// get the vector of combinations for the algorithm
// dummy here (not needed for objects templates), redefined for (pre)algorithms
std::vector<CombinationsInCond> 
    L1GlobalTriggerConditions::getCombinationVector() {
        
    std::vector<CombinationsInCond> dummy;
    return dummy;
            
}

// get the vector of object types for the algorithm
// dummy here (not needed for objects templates), redefined for (pre)algorithms
std::vector<ObjectTypeInCond> 
    L1GlobalTriggerConditions::getObjectTypeVector() {
        
    std::vector<ObjectTypeInCond> dummy;
    return dummy;
            
}

