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


// set the condition maps
void L1GtTriggerMenu::setGtConditionMap(const std::vector<ConditionMap>& condMap)
{
    m_conditionMap = condMap;
}

// set the algorithm map
void L1GtTriggerMenu::setGtAlgorithmMap(const AlgorithmMap& algoMap)
{
    m_algorithmMap = algoMap;
}


// print the trigger menu (bit number, algorithm name, logical expression)
void L1GtTriggerMenu::print(std::ostream& myCout, int& printVerbosity ) const
{


    // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
    // both algorithm and bit numbers are unique
    std::map<int, L1GtAlgorithm> algoBitToAlgo;
    typedef std::map<int, L1GtAlgorithm>::const_iterator CItBit;


    for (CItAlgo
            itAlgo  = m_algorithmMap.begin();
            itAlgo != m_algorithmMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second).algoBitNumber();
        algoBitToAlgo[bitNumber] = itAlgo->second;
    }



    switch (printVerbosity) {

        case 0: {

                // header for printing algorithms

                myCout
                << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
                << "Bit Number " << " Algorithm Name "
                << std::endl;

                for (CItBit
                        itBit  = algoBitToAlgo.begin();
                        itBit != algoBitToAlgo.end(); itBit++) {

                    int bitNumber = itBit->first;
                    std::string aName = (itBit->second).algoName();

                    myCout
                    << std::setw(6) << bitNumber << "       "
                    << aName
                    << std::endl;
                }
            }
            break;

        case 1: {

                // header for printing algorithms

                myCout
                << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
                << "Bit Number " << " Algorithm Name " << "\n  Logical Expresssion \n"
                << std::endl;

                for (CItBit
                        itBit  = algoBitToAlgo.begin();
                        itBit != algoBitToAlgo.end(); itBit++) {

                    int bitNumber = itBit->first;
                    std::string aName = (itBit->second).algoName();
                    std::string aLogicalExpression =
                        (itBit->second).algoLogicalExpression();

                    myCout
                    << std::setw(6) << bitNumber << "       "
                    << aName
                    << "\n  Logical expression: " << aLogicalExpression << "\n"
                    << std::endl;
                }
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
