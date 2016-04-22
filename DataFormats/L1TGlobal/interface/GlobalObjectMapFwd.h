#ifndef L1GlobalTrigger_L1TGtObjectMapFwd_h
#define L1GlobalTrigger_L1TGtObjectMapFwd_h

/**
 * \class GlobalObjectMap
 * 
 * 
 * Description: group typedefs used by GlobalObjectMap.  
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
#include "DataFormats/L1TGlobal/interface/GlobalObject.h"

// forward declarations

/// typedefs

/// list of object indices corresponding to a condition evaluated to true
typedef std::vector<int> SingleCombInCond;

/// all the object combinations evaluated to true in the condition
typedef std::vector<SingleCombInCond> CombinationsInCond;

typedef std::vector<l1t::GlobalObject> ObjectTypeInCond;
//typedef std::vector<int> ObjectTypeInCond;

#endif /* L1GlobalTrigger_L1TGtObjectMapFwd_h */
