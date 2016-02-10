#ifndef L1GlobalTrigger_L1TGtObjectMapFwd_h
#define L1GlobalTrigger_L1TGtObjectMapFwd_h

/**
 * \class L1TGtObjectMap
 * 
 * 
 * Description: group typedefs used by L1TGtObjectMap.  
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
#include "L1Trigger/L1TGlobal/interface/L1TGtObject.h"

// forward declarations

/// typedefs

/// list of object indices corresponding to a condition evaluated to true
typedef std::vector<int> SingleCombInCond;

/// all the object combinations evaluated to true in the condition
typedef std::vector<SingleCombInCond> CombinationsInCond;

typedef std::vector<l1t::L1TGtObject> ObjectTypeInCond;
//typedef std::vector<int> ObjectTypeInCond;

#endif /* L1GlobalTrigger_L1TGtObjectMapFwd_h */
