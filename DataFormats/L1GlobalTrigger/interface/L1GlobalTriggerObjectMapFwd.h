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
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <vector>

// user include files
//   base class
// forward declarations

/// typedefs

/// list of object indices corresponding to a condition evaluated to true
typedef std::vector<int> SingleCombInCond;

/// all the object combinations evaluated to true in the condition
typedef std::vector<SingleCombInCond> CombinationsInCond;


#endif /* L1GlobalTrigger_L1GlobalTriggerObjectMapFwd_h */
