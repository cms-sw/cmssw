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
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

// system include files
#include <iomanip>


// user include files
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"


// forward declarations

// constructor
L1GtTriggerMenu::L1GtTriggerMenu() {
    // empty
}

// destructor
L1GtTriggerMenu::~L1GtTriggerMenu() {

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    for (std::vector<ConditionMap>::iterator 
        itCondOnChip = m_conditionMap.begin(); itCondOnChip != m_conditionMap.end(); itCondOnChip++) {

        for (ItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {

            if (itCond->second != 0) {
                delete itCond->second;
            }
            itCond->second = 0;

        }

        itCondOnChip->clear();

    }

    for (ItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {

        if (itAlgo->second != 0) {
            delete itAlgo->second;
        }
        itAlgo->second = 0;

    }

    m_algorithmMap.clear();
}

/// set the trigger menu name
void L1GtTriggerMenu::setGtTriggerMenuName(const std::string& menuName) {
    m_triggerMenuName = menuName;
}

// set the condition maps
void L1GtTriggerMenu::setGtConditionMap(const std::vector<ConditionMap>& condMap) {
    m_conditionMap = condMap;
}

// set the algorithm map
void L1GtTriggerMenu::setGtAlgorithmMap(const AlgorithmMap& algoMap) {
    m_algorithmMap = algoMap;
}

// print the trigger menu (bit number, algorithm name, logical expression)
void L1GtTriggerMenu::print(std::ostream& myCout, int& printVerbosity) const {

    // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
    // both algorithm and bit numbers are unique
    std::map<int, L1GtAlgorithm*> algoBitToAlgo;
    typedef std::map<int, L1GtAlgorithm*>::const_iterator CItBit;

    for (CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second)->algoBitNumber();
        algoBitToAlgo[bitNumber] = itAlgo->second;
    }

    switch (printVerbosity) {

        case 0: {

            // header for printing algorithms

            myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "L1 Trigger Menu Name: " << m_triggerMenuName << "\n\n"
            << "Bit Number " << " Algorithm Name " << std::endl;

            for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {

                int bitNumber = itBit->first;
                std::string aName = (itBit->second)->algoName();

                myCout << std::setw(6) << bitNumber << "       " << aName << std::endl;
            }
        }
            break;

        case 1: {

            // header for printing algorithms

            myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "L1 Trigger Menu Name: " << m_triggerMenuName << "\n\n"
            << "Bit Number " << " Algorithm Name " << "\n  Logical Expresssion \n" << std::endl;

            for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {

                int bitNumber = itBit->first;
                std::string aName = (itBit->second)->algoName();
                std::string aLogicalExpression = (itBit->second)->algoLogicalExpression();

                myCout << std::setw(6) << bitNumber << "       " << aName
                    << "\n  Logical expression: " << aLogicalExpression << "\n" << std::endl;
            }
        }
            break;

        case 2: {

            // header for printing algorithms

            myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "L1 Trigger Menu Name: " << m_triggerMenuName << "\n\n"
            << std::endl;

            for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {
                (itBit->second)->print(myCout);
            }
        }
            break;

        default: {
            myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "Verbosity level: " << printVerbosity << " not implemented.\n\n"
            << std::endl;
        }
            break;
    }

}

// get the result for algorithm with name algName
// use directly the format of decisionWord (no typedef) 
const bool L1GtTriggerMenu::gtAlgorithmResult(const std::string& algName,
        const std::vector<bool>& decWord) const {

    bool algResult = false;

    CItAlgo itAlgo = m_algorithmMap.find(algName);
    if (itAlgo != m_algorithmMap.end()) {
        int bitNumber = (itAlgo->second)->algoBitNumber();
        algResult = decWord.at(bitNumber);
        return algResult;
    }

    // return false if the algorithm name is not found in the menu
    // TODO throw exception or LogInfo would be better - but the class is used in 
    // XDAQ Trigger Supervisor (outside CMSSW) hence no CMSSW dependence
    // is allowed here...

    return false;

}
