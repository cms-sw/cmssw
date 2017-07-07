/**
 * \class L1GtTechnicalTriggerRecord
 * 
 * 
 * Description: technical trigger input record for L1 Global Trigger.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations


// constructor(s)
L1GtTechnicalTriggerRecord::L1GtTechnicalTriggerRecord()
{
    // empty    
}

// destructor
L1GtTechnicalTriggerRecord::~L1GtTechnicalTriggerRecord()
{
    // empty
}

// methods

// return the technical trigger with ttName and bxInEvent
const L1GtTechnicalTrigger* L1GtTechnicalTriggerRecord::getTechnicalTrigger(
        const std::string& ttName, const int bxInEventVal) const
{

    for (std::vector<L1GtTechnicalTrigger>::const_iterator itObj =
            m_gtTechnicalTrigger.begin(); itObj != m_gtTechnicalTrigger.end(); ++itObj) {

        if ((itObj->gtTechnicalTriggerName() == ttName) 
                && (itObj->bxInEvent() == bxInEventVal)) {

            return &((*itObj));
        }

    }

    // no ttName found, return zero pointer!
    edm::LogError("L1GtTechnicalTriggerRecord")
            << "\n\n  ERROR: The requested technical trigger name = " << ttName
            << " with bxInEvent = " << bxInEventVal << "\n  does not exist."
            << "\n  Returning zero pointer for getTechnicalTrigger\n\n"
            << std::endl;

    return nullptr;

}

/// return the technical trigger for ttBitNumber and bxInEvent
const L1GtTechnicalTrigger* L1GtTechnicalTriggerRecord::getTechnicalTrigger(
        const unsigned int ttBitNumber, const int bxInEventVal) const
{

    for (std::vector<L1GtTechnicalTrigger>::const_iterator itObj =
            m_gtTechnicalTrigger.begin(); itObj != m_gtTechnicalTrigger.end(); ++itObj) {

        if ((itObj->gtTechnicalTriggerBitNumber() == ttBitNumber)
                && (itObj->bxInEvent() == bxInEventVal)) {

            return &((*itObj));
        }

    }

    // no ttBitNumber && bxInEventVal found, return zero pointer!
    edm::LogError("L1GtTechnicalTriggerRecord")
            << "\n\n  ERROR: The requested technical trigger with bit number = "
            << ttBitNumber << " and with bxInEvent = " << bxInEventVal
            << "\n  does not exist."
            << "\n  Returning zero pointer for getTechnicalTrigger\n\n"
            << std::endl;

    return nullptr;

}

