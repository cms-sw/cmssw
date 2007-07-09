/**
 * \class L1GlobalTriggerObjectMapRecord
 * 
 * 
 * Description: see header file.  
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

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// system include files
#include <string>
#include <vector>

#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

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
    std::string algoNameVal, std::string condNameVal) const
{

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
            itObj != m_gtObjectMap.end(); ++itObj) {

        if ( (*itObj).algoName() == algoNameVal ) {

            L1GtLogicParser logicParser( (*itObj).algoLogicalExpression(),
                                         (*itObj).algoNumericalExpression());
            int conditionIndexVal = logicParser.operandIndex(condNameVal);

            return &((*itObj).combinationVector().at(conditionIndexVal));
        }
    }

    // no (algoName, condName) found, return zero pointer!
    edm::LogError("L1GlobalTriggerObjectMapRecord")
    << "\n\n  ERROR: The requested \n    (algorithm name, condition name) = ("
    << algoNameVal << ", " << condNameVal
    << ") \n  does not exists in the trigger menu."
    << "\n  Returning zero pointer for getCombinationsInCond\n\n"
    << std::endl;

    return 0;

}

// return all the combinations passing the requirements imposed in condition condNameVal
// from algorithm with bit number algoBitNumberVal
const CombinationsInCond* L1GlobalTriggerObjectMapRecord::getCombinationsInCond(
    int algoBitNumberVal, std::string condNameVal) const
{

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
            itObj != m_gtObjectMap.end(); ++itObj) {

        if ( (*itObj).algoBitNumber() == algoBitNumberVal ) {
            L1GtLogicParser logicParser( (*itObj).algoLogicalExpression(),
                                         (*itObj).algoNumericalExpression());
            int conditionIndexVal = logicParser.operandIndex(condNameVal);

            return &((*itObj).combinationVector().at(conditionIndexVal));
        }
    }

    // no (algoBitNumber, condName) found, return zero pointer!
    edm::LogError("L1GlobalTriggerObjectMapRecord")
    << "\n\n  ERROR: The requested \n    (algorithm bit number, condition name) = ("
    << algoBitNumberVal << ", " << condNameVal
    << ") \n  does not exists in the trigger menu."
    << "\n  Returning zero pointer for getCombinationsInCond\n\n"
    << std::endl;

    return 0;

}

// return the result for the condition condNameVal
// from algorithm with name algoNameVal
const bool L1GlobalTriggerObjectMapRecord::getConditionResult(
    std::string algoNameVal, std::string condNameVal) const
{

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
            itObj != m_gtObjectMap.end(); ++itObj) {

        if ( (*itObj).algoName() == algoNameVal ) {

            L1GtLogicParser logicParser( (*itObj).algoLogicalExpression(),
                                         (*itObj).algoNumericalExpression());
            return logicParser.operandResult(condNameVal);
        }
    }

    // no (algoName, condName) found, return false!
    edm::LogError("L1GlobalTriggerObjectMapRecord")
    << "\n\n  ERROR: The requested \n    (algorithm name, condition name) = ("
    << algoNameVal << ", " << condNameVal
    << ") \n  does not exists in the trigger menu."
    << "\n  Returning false for condition result! Unknown result, in fact!\n\n"
    << std::endl;

    return false;

}

// return the result for the condition condNameVal
// from algorithm with bit number algoBitNumberVal
const bool L1GlobalTriggerObjectMapRecord::getConditionResult(
    int algoBitNumberVal, std::string condNameVal) const
{
    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
            itObj != m_gtObjectMap.end(); ++itObj) {

        if ( (*itObj).algoBitNumber() == algoBitNumberVal ) {
            L1GtLogicParser logicParser( (*itObj).algoLogicalExpression(),
                                         (*itObj).algoNumericalExpression());

            return logicParser.operandResult(condNameVal);
        }
    }

    // no (algoBitNumber, condName) found, return false!
    edm::LogError("L1GlobalTriggerObjectMapRecord")
    << "\n\n  ERROR: The requested \n    (algorithm bit number, condition name) = ("
    << algoBitNumberVal << ", " << condNameVal
    << ") \n  does not exists in the trigger menu."
    << "\n  Returning false for condition result! Unknown result, in fact!\n\n"
    << std::endl;

    return false;

}
