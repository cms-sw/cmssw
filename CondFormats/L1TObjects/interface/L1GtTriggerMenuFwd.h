#ifndef CondFormats_L1TObjects_L1GtTriggerMenuFwd_h
#define CondFormats_L1TObjects_L1GtTriggerMenuFwd_h

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
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <string>
#include <map>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

/// map containing the conditions
typedef std::map<std::string, L1GtCondition*> ConditionsMap;

/// map containing the algorithms
typedef std::map<std::string, L1GtAlgorithm*> AlgorithmsMap;

/// constant iterator through map containing the conditions
typedef ConditionsMap::const_iterator CItCond;

/// constant iterator through map containing the algorithms
typedef AlgorithmsMap::const_iterator CItAlgo;


#endif /*CondFormats_L1TObjects_L1GtTriggerMenuFwd_h*/
