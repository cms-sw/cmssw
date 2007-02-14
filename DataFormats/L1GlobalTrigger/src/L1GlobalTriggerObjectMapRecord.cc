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
 * $Date$
 * $Revision$
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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations


// constructor(s)
L1GlobalTriggerObjectMapRecord::L1GlobalTriggerObjectMapRecord()
{}

// destructor
L1GlobalTriggerObjectMapRecord::~L1GlobalTriggerObjectMapRecord()
{}

// methods

// return all the combinations passing the requirements imposed in condition condNameVal
// from algorithm algoNameVal
const CombinationsInCond* L1GlobalTriggerObjectMapRecord::getCombinationsInCond(
    std::string algoNameVal, std::string condNameVal)
{

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
            itObj != m_gtObjectMap.end(); ++itObj) {

        if ( (*itObj).algoName() == algoNameVal ) {
            int conditionIndex = 0; // FIXME implement simple parser for logical expression

            return &((*itObj).combinationVector().at(conditionIndex));
        }
    }

    // no (algoName, condName) found, return zero pointer!

    edm::LogError("L1GlobalTriggerObjectMapRecord")
    << "\n  ERROR: The requested (algorithm name, condition name) = ("
    << algoNameVal << ", " << condNameVal 
    << ") does not exists in the trigger menu." 
    << "  Returning zero pointer for getCombinationsInCond" 
    << std::endl;

    return 0;

}

// return all the combinations passing the requirements imposed in condition condNameVal
// from algorithm with bit number algoBitNumberVal
const CombinationsInCond* L1GlobalTriggerObjectMapRecord::getCombinationsInCond(
    int algoBitNumberVal, std::string condNameVal)
{

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
            itObj != m_gtObjectMap.end(); ++itObj) {

        if ( (*itObj).algoBitNumber() == algoBitNumberVal ) {
            int conditionIndex = 0; // FIXME implement simple parser for logical expression

            return &((*itObj).combinationVector().at(conditionIndex));
        }
    }

    // no (algoBitNumber, condName) found, return zero pointer!
    edm::LogError("L1GlobalTriggerObjectMapRecord")
    << "\n  ERROR: The requested (algorithm bit number, condition name) = ("
    << algoBitNumberVal << ", " << condNameVal 
    << ") does not exists in the trigger menu." 
    << "  Returning zero pointer for getCombinationsInCond" 
    << std::endl;

    return 0;

}
