#ifndef DataFormats_L1TGlobal_GlobalObjectMapFwd_h
#define DataFormats_L1TGlobal_GlobalObjectMapFwd_h

/**
 * \class GlobalObjectMapFwd
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
#include <utility>
#include <vector>

// user include files
#include "DataFormats/L1TGlobal/interface/GlobalObject.h"

// forward declarations

/// typedefs
typedef int16_t L1TObjBxIndexType;
typedef int L1TObjIndexType;

/// list of object indices:bx pairs corresponding to a condition evaluated to true
typedef std::vector<std::pair<L1TObjBxIndexType, L1TObjIndexType>> SingleCombWithBxInCond;

/// all the object combinations evaluated to true in the condition (object indices + BX indices)
typedef std::vector<SingleCombWithBxInCond> CombinationsWithBxInCond;

typedef std::vector<l1t::GlobalObject> L1TObjectTypeInCond;

#endif
