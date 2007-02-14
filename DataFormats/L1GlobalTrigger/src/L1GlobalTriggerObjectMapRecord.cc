/**
 * \class L1GlobalTriggerObjectMapRecord
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// system include files
#include <string>
#include <vector>

#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

// forward declarations


// constructor(s)
L1GlobalTriggerObjectMapRecord::L1GlobalTriggerObjectMapRecord()
{}

// destructor
L1GlobalTriggerObjectMapRecord::~L1GlobalTriggerObjectMapRecord()
{}

// methods

// return all the combinations passing the requirements imposed in condition condName
// from algorithm algoName
const CombinationsInCond* L1GlobalTriggerObjectMapRecord::getCombinationsInCond(
    std::string algoNameVal, std::string condNameVal)
{

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_GtObjectMap.begin(); 
        itObj != m_GtObjectMap.end(); ++itObj) {
        
        if ( (*itObj).algoName() == algoNameVal ) {
            int conditionIndex = 0; // FIXME

           return &((*itObj).combinationVector().at(conditionIndex));
        }               
    }
    
    return 0;
    
}

