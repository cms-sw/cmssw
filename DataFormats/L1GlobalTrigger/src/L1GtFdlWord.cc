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

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtFdlWord::L1GtFdlWord() {

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    // technical triggers std::vector<bool>
    m_gtTechnicalTrigger.reserve(L1GlobalTriggerReadoutRecord::NumberTechnicalTriggers);
    m_gtTechnicalTrigger.assign(L1GlobalTriggerReadoutRecord::NumberTechnicalTriggers, false);
    
    // decision word  std::vector<bool>      
    m_gtDecision.reserve(L1GlobalTriggerReadoutRecord::NumberPhysTriggers);
    m_gtDecision.assign(L1GlobalTriggerReadoutRecord::NumberPhysTriggers, false);

    // extended decision word  std::vector<bool>      
    m_gtDecisionExtended.reserve(L1GlobalTriggerReadoutRecord::NumberPhysTriggersExtended);
    m_gtDecisionExtended.assign(L1GlobalTriggerReadoutRecord::NumberPhysTriggersExtended, false);

    m_finalOR = 0;
    m_localBxNr = 0;

}

// constructor from unpacked values;
L1GtFdlWord::L1GtFdlWord(
    uint16_t boardIdValue,
    uint16_t bxInEventValue,
    uint16_t bxNrValue,
    uint32_t eventNrValue,
    L1GlobalTriggerReadoutRecord::TechnicalTriggerWord gtTechnicalTriggerValue,
    L1GlobalTriggerReadoutRecord::DecisionWord gtDecisionValue,
    L1GlobalTriggerReadoutRecord::DecisionWordExtended gtDecisionExtendedValue,
    uint16_t finalORValue,
    uint16_t localBxNrValue
    ) 
{

    m_boardId = boardIdValue;
    m_bxInEvent = bxInEventValue;
    m_bxNr = bxNrValue;
    m_eventNr = eventNrValue;
    m_gtTechnicalTrigger = gtTechnicalTriggerValue;
    m_gtDecision = gtDecisionValue;
    m_gtDecisionExtended = gtDecisionExtendedValue;
    m_finalOR = finalORValue;
    m_localBxNr = localBxNrValue;
    
}

// destructor
L1GtFdlWord::~L1GtFdlWord() {
}

// methods


// static class members


