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

L1GtTriggerMenu::L1GtTriggerMenu(
        const std::string& triggerMenuNameVal,
        const unsigned int numberConditionChips,
        const std::vector<std::vector<L1GtMuonTemplate> >& vecMuonTemplateVal,
        const std::vector<std::vector<L1GtCaloTemplate> >& vecCaloTemplateVal,
        const std::vector<std::vector<L1GtEnergySumTemplate> >& vecEnergySumTemplateVal,
        const std::vector<std::vector<L1GtJetCountsTemplate> >& vecJetCountsTemplateVal,
        const std::vector<std::vector<L1GtCorrelationTemplate> >& vecCorrelationTemplateVal,
        const std::vector<std::vector<L1GtMuonTemplate> >& corMuonTemplateVal,
        const std::vector<std::vector<L1GtCaloTemplate> >& corCaloTemplateVal,
        const std::vector<std::vector<L1GtEnergySumTemplate> >& corEnergySumTemplateVal

) :
    m_triggerMenuName(triggerMenuNameVal),
            m_vecMuonTemplate(vecMuonTemplateVal),
            m_vecCaloTemplate(vecCaloTemplateVal),
            m_vecEnergySumTemplate(vecEnergySumTemplateVal),
            m_vecJetCountsTemplate(vecJetCountsTemplateVal),
            m_vecCorrelationTemplate(vecCorrelationTemplateVal),
            m_corMuonTemplate(corMuonTemplateVal),
            m_corCaloTemplate(corCaloTemplateVal),
            m_corEnergySumTemplate(corEnergySumTemplateVal) {

    m_conditionMap.resize(numberConditionChips);
    buildGtConditionMap();

}

// destructor
L1GtTriggerMenu::~L1GtTriggerMenu() {

    // loop over condition maps (one map per condition chip)
    for (std::vector<ConditionMap>::iterator 
        itCondOnChip = m_conditionMap.begin(); itCondOnChip != m_conditionMap.end(); itCondOnChip++) {

        itCondOnChip->clear();

    }

    m_algorithmMap.clear();
}

// set the condition maps
void L1GtTriggerMenu::setGtConditionMap(const std::vector<ConditionMap>& condMap) {
    m_conditionMap = condMap;
}

// build the condition maps
void L1GtTriggerMenu::buildGtConditionMap() {

    int chipNr = -1;
    
    for (std::vector<std::vector<L1GtMuonTemplate> >::iterator 
            itCondOnChip = m_vecMuonTemplate.begin();
            itCondOnChip != m_vecMuonTemplate.end();
            itCondOnChip++) {

        chipNr++;
        
        for (std::vector<L1GtMuonTemplate>::iterator 
                itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
                itCond++) {
            
            (m_conditionMap[chipNr])[itCond->condName()] = &(*itCond);
        }
    }

    chipNr = -1;
    for (std::vector<std::vector<L1GtCaloTemplate> >::iterator 
            itCondOnChip = m_vecCaloTemplate.begin();
            itCondOnChip != m_vecCaloTemplate.end();
            itCondOnChip++) {

        chipNr++;
        
        for (std::vector<L1GtCaloTemplate>::iterator 
                itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
                itCond++) {
            
            (m_conditionMap[chipNr])[itCond->condName()] = &(*itCond);
        }
    }
    
    chipNr = -1;
    for (std::vector<std::vector<L1GtEnergySumTemplate> >::iterator 
            itCondOnChip = m_vecEnergySumTemplate.begin();
            itCondOnChip != m_vecEnergySumTemplate.end();
            itCondOnChip++) {

        chipNr++;
        
        for (std::vector<L1GtEnergySumTemplate>::iterator 
                itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
                itCond++) {
            
            (m_conditionMap[chipNr])[itCond->condName()] = &(*itCond);
        }
    }
    
    chipNr = -1;
    for (std::vector<std::vector<L1GtJetCountsTemplate> >::iterator 
            itCondOnChip = m_vecJetCountsTemplate.begin();
            itCondOnChip != m_vecJetCountsTemplate.end();
            itCondOnChip++) {

        chipNr++;
        
        for (std::vector<L1GtJetCountsTemplate>::iterator 
                itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
                itCond++) {
            
            (m_conditionMap[chipNr])[itCond->condName()] = &(*itCond);
        }
    }
    
    
    chipNr = -1;
    for (std::vector<std::vector<L1GtCorrelationTemplate> >::iterator 
            itCondOnChip = m_vecCorrelationTemplate.begin();
            itCondOnChip != m_vecCorrelationTemplate.end();
            itCondOnChip++) {

        chipNr++;
        
        for (std::vector<L1GtCorrelationTemplate>::iterator 
                itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
                itCond++) {
            
            (m_conditionMap[chipNr])[itCond->condName()] = &(*itCond);
        }
    }
    
    
    
    
}

// set the trigger menu name
void L1GtTriggerMenu::setGtTriggerMenuName(const std::string& menuName) {
    m_triggerMenuName = menuName;
}

// get / set the vectors containing the conditions
void L1GtTriggerMenu::setVecMuonTemplate(
        const std::vector<std::vector<L1GtMuonTemplate> >& vecMuonTempl) {
    
    m_vecMuonTemplate = vecMuonTempl;
}

void L1GtTriggerMenu::setVecCaloTemplate(
        const std::vector<std::vector<L1GtCaloTemplate> >& vecCaloTempl) {
    
    m_vecCaloTemplate = vecCaloTempl;
}

void L1GtTriggerMenu::setVecEnergySumTemplate(
        const std::vector<std::vector<L1GtEnergySumTemplate> >& vecEnergySumTempl) {
    
    m_vecEnergySumTemplate = vecEnergySumTempl;
}

void L1GtTriggerMenu::setVecJetCountsTemplate(
        const std::vector<std::vector<L1GtJetCountsTemplate> >& vecJetCountsTempl) {
    
    m_vecJetCountsTemplate = vecJetCountsTempl;
}

void L1GtTriggerMenu::setVecCorrelationTemplate(
        const std::vector<std::vector<L1GtCorrelationTemplate> >& vecCorrelationTempl) {
    
    m_vecCorrelationTemplate = vecCorrelationTempl;
}

// set the vectors containing the conditions for correlation templates
void L1GtTriggerMenu::setCorMuonTemplate(
        const std::vector<std::vector<L1GtMuonTemplate> >& corMuonTempl) {
    
    m_corMuonTemplate = corMuonTempl;
}

void L1GtTriggerMenu::setCorCaloTemplate(
        const std::vector<std::vector<L1GtCaloTemplate> >& corCaloTempl) {
    
    m_corCaloTemplate = corCaloTempl;
}

void L1GtTriggerMenu::setCorEnergySumTemplate(
        const std::vector<std::vector<L1GtEnergySumTemplate> >& corEnergySumTempl) {
    
    m_corEnergySumTemplate = corEnergySumTempl;
}



// set the algorithm map
void L1GtTriggerMenu::setGtAlgorithmMap(const AlgorithmMap& algoMap) {
    m_algorithmMap = algoMap;
}

// print the trigger menu (bit number, algorithm name, logical expression)
void L1GtTriggerMenu::print(std::ostream& myCout, int& printVerbosity) const {

    // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
    // both algorithm and bit numbers are unique
    std::map<int, const L1GtAlgorithm*> algoBitToAlgo;
    typedef std::map<int, const L1GtAlgorithm*>::const_iterator CItBit;

    for (CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second).algoBitNumber();
        algoBitToAlgo[bitNumber] = &(itAlgo->second);
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
            

            myCout << "\nNumber of condition chips: " << m_conditionMap.size() << "\n"
            << std::endl;

            int chipNr = -1;
            int totalNrConditions = 0;
            
            for (std::vector<ConditionMap>::const_iterator 
                    itCondOnChip = m_conditionMap.begin(); 
                    itCondOnChip != m_conditionMap.end(); itCondOnChip++) {

                chipNr++;
                
                int condMapSize = itCondOnChip->size();
                totalNrConditions += condMapSize;
                
                myCout << "\nTotal number of conditions on condition chip " << chipNr 
                << ": " << condMapSize
                << " conditions.\n" << std::endl;

                for (CItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
                    itCond++) {

                    (itCond->second)->print(myCout);
                
                }

            }
            
            myCout << "\nTotal number of conditions on all condition chips: " 
            << totalNrConditions << "\n" 
            << std::endl;
            
                                
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
        int bitNumber = (itAlgo->second).algoBitNumber();
        algResult = decWord.at(bitNumber);
        return algResult;
    }

    // return false if the algorithm name is not found in the menu
    // TODO throw exception or LogInfo would be better - but the class is used in 
    // XDAQ Trigger Supervisor (outside CMSSW) hence no CMSSW dependence
    // is allowed here...

    return false;

}


