#ifndef CondFormats_L1TObjects_L1uGtTriggerMenuFwd_h
#define CondFormats_L1TObjects_L1uGtTriggerMenuFwd_h

/**
 * \class L1GtTriggerMenu 
 * 
 * Description: forward header for L1 Global Trigger menu.  
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
#include <string>
#include <map>

// user include files
#include "CondFormats/L1TObjects/interface/L1uGtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

/// map containing the conditions
namespace l1t {
typedef std::map<std::string, L1uGtCondition*> ConditionMap;

/// map containing the algorithms
typedef std::map<std::string, L1GtAlgorithm> AlgorithmMap;

/// iterators through map containing the conditions
typedef ConditionMap::const_iterator CItCond;
typedef ConditionMap::iterator ItCond;

/// iterators through map containing the algorithms
typedef AlgorithmMap::const_iterator CItAlgo;
typedef AlgorithmMap::iterator ItAlgo;

}
#endif /*CondFormats_L1TObjects_L1uGtTriggerMenuFwd_h*/
