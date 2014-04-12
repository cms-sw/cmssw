#ifndef L1GlobalTrigger_L1GlobalTriggerObjectMapFwd_h
#define L1GlobalTrigger_L1GlobalTriggerObjectMapFwd_h

/**
 * \class L1GlobalTriggerObjectMap
 * 
 * 
 * Description: group typedefs used by L1GlobalTriggerObjectMap.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

/// typedefs

/// list of object indices corresponding to a condition evaluated to true
typedef std::vector<int> SingleCombInCond;

/// all the object combinations evaluated to true in the condition
typedef std::vector<SingleCombInCond> CombinationsInCond;

typedef std::vector<L1GtObject> ObjectTypeInCond;

#endif /* L1GlobalTrigger_L1GlobalTriggerObjectMapFwd_h */
