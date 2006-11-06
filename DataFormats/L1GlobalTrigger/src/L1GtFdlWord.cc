/**
 * \class L1GtFdlWord
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
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

// system include files
#include <iostream>
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtFdlWord::L1GtFdlWord() {

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    // technical triggers std::vector<bool>
    m_gtTechnicalTriggerWord.reserve(L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers);
    m_gtTechnicalTriggerWord.assign(L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers, false);
    
    // decision word  std::vector<bool>      
    m_gtDecisionWord.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_gtDecisionWord.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers, false);

    // extended decision word  std::vector<bool>      
    m_gtDecisionWordExtended.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended);
    m_gtDecisionWordExtended.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended, false);

    m_finalOR = 0;
    m_localBxNr = 0;

}

// constructor from unpacked values;
L1GtFdlWord::L1GtFdlWord(
    uint16_t boardIdValue,
    uint16_t bxInEventValue,
    uint16_t bxNrValue,
    uint32_t eventNrValue,
    L1GlobalTriggerReadoutSetup::TechnicalTriggerWord gtTechnicalTriggerWordValue,
    L1GlobalTriggerReadoutSetup::DecisionWord gtDecisionWordValue,
    L1GlobalTriggerReadoutSetup::DecisionWordExtended gtDecisionWordExtendedValue,
    uint16_t finalORValue,
    uint16_t localBxNrValue
    ) 
{

    m_boardId = boardIdValue;
    m_bxInEvent = bxInEventValue;
    m_bxNr = bxNrValue;
    m_eventNr = eventNrValue;
    m_gtTechnicalTriggerWord = gtTechnicalTriggerWordValue;
    m_gtDecisionWord = gtDecisionWordValue;
    m_gtDecisionWordExtended = gtDecisionWordExtendedValue;
    m_finalOR = finalORValue;
    m_localBxNr = localBxNrValue;
    
}

// destructor
L1GtFdlWord::~L1GtFdlWord() {
}

// equal operator
bool L1GtFdlWord::operator==(const L1GtFdlWord& result) const {

    if (m_boardId != m_boardId) return false;
    if (m_bxInEvent != m_bxInEvent) return false;
    if (m_bxNr != m_bxNr) return false;
    if (m_eventNr != m_eventNr) return false;
    if (m_gtTechnicalTriggerWord != m_gtTechnicalTriggerWord) return false;
    if (m_gtDecisionWord != m_gtDecisionWord) return false;
    if (m_gtDecisionWordExtended != m_gtDecisionWordExtended) return false;
    if (m_finalOR != m_finalOR) return false;
    if (m_localBxNr != m_localBxNr) return false;

    // all members identical
    return true;
    
}

// unequal operator
bool L1GtFdlWord::operator!=(const L1GtFdlWord& result) const{
    
    return !( result == *this);
    
}


// methods

// print GT technical trigger word in bitset style
//    depend on the type of L1GlobalTriggerReadoutSetup::TechnicalTriggerWord
//    this version: <vector<bool>
void L1GtFdlWord::printGtTechnicalTriggerWord(std::ostream& myCout) const {
    
    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtTechnicalTriggerWord.rbegin(); 
        ritBit != m_gtTechnicalTriggerWord.rend(); ++ritBit) {
        
        myCout << (*ritBit ? '1' : '0');
                
    }      
    
    
} 

// print GT decision word in bitset style
//    depend on the type of L1GlobalTriggerReadoutSetup::DecisionWord
//    this version: <vector<bool>
void L1GtFdlWord::printGtDecisionWord(std::ostream& myCout) const {
    
    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWord.rbegin(); 
        ritBit != m_gtDecisionWord.rend(); ++ritBit) {
        
        myCout << (*ritBit ? '1' : '0');
                
    }      
    
    
} 


// reset the content of a L1GtFdlWord
void L1GtFdlWord::reset() {
    
    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    // technical triggers std::vector<bool>
    m_gtTechnicalTriggerWord.assign(L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers, false);
    
    // decision word  std::vector<bool>      
    m_gtDecisionWord.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers, false);

    // extended decision word  std::vector<bool>      
    m_gtDecisionWordExtended.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggersExtended, false);

    m_finalOR = 0;
    m_localBxNr = 0;
    
}

// static class members


