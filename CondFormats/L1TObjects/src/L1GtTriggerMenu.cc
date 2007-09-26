/**
 * \class L1GtTriggerMenu
 * 
 * 
 * Description: L1 trigger menu.  
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
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

// system include files
#include <ostream>
#include <iomanip>

#include <boost/cstdint.hpp>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

// forward declarations

// constructor
L1GtTriggerMenu::L1GtTriggerMenu()
{
    // empty
}

// destructor
L1GtTriggerMenu::~L1GtTriggerMenu()
{
    // empty
}


// print the trigger menu (bit number, algorithm name, logical expression)
void L1GtTriggerMenu::print(std::ostream& myCout, int& printVerbosity ) const
{

    switch (printVerbosity) {

        case 0: {

                // header for printing algorithms

                myCout
                << "\n L1 Trigger Menu - short printing\n"
                << "  Bit Number " << " Algorithm Name " << "Logical Expresssion \n"
                << std::endl;

                for (CItAlgo itAlgo  = m_algorithmsMap.begin();
                        itAlgo != m_algorithmsMap.end(); itAlgo++) {

                    int bitNumber = 0; // FIXME
                    std::string algoName = itAlgo->first;
                    std::string algoLogicalExpression = ""; // FIXME

                    myCout
                    << algoName
                    << "  " << bitNumber
                    << " = " << algoLogicalExpression
                    << std::endl;
                }
            }
            break;

        case 1: {

                // more verbose
            }
            break;

        case 2: {

                // more more verbose
            }
            break;

        default: {
                // write some informative message
            }
            break;
    }


}

// clearMaps - delete all conditions in the maps and clear the maps.
void L1GtTriggerMenu::clearMaps()
{

    // FIXME

}


// insertConditionIntoMap - safe insert of condition into condition map.
// if the condition name already exists, do not insert it and return false
bool L1GtTriggerMenu::insertConditionIntoMap(L1GtCondition* cond, int chipNr,
        std::ostream& myCout)
{

    // FIXME
    return true;
}


// insertAlgorithmIntoMap - safe insert of Algorithm into Algorithm map.
// if the condition name already exists, do not insert it and return false
bool L1GtTriggerMenu::insertAlgoIntoMap(L1GtAlgorithm* algo,
                                        AlgorithmsMap* insertMap, ConditionsMap* operandMap,
                                        int chipNr, std::ostream& myCout)
{

    // FIXME
    return true;
}
