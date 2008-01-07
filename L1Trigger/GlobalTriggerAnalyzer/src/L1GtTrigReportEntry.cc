/**
 * \class L1GtTrigReportEntry
 * 
 * 
 * Description: L1 Trigger report.  
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"

// system include files
#include <string>

// user include files

// constructor(s)
L1GtTrigReportEntry::L1GtTrigReportEntry(const std::string& menuName, const std::string& algName,
    const int prescaleFactor, const int triggerMask) {

    m_triggerMenuName = menuName;
    m_algoName = algName;
    m_prescaleFactor = prescaleFactor;
    m_triggerMask = triggerMask;

    // counters

    m_nrEventsAccept = 0;
    m_nrEventsReject = 0;
    m_nrEventsError = 0;
}

// destructor
L1GtTrigReportEntry::~L1GtTrigReportEntry() {

    //empty

}

// assignment operator
L1GtTrigReportEntry::L1GtTrigReportEntry& L1GtTrigReportEntry::operator=(
    const L1GtTrigReportEntry& repEntry)
{

    if ( this != &repEntry ) {

        m_triggerMenuName = repEntry.gtTriggerMenuName();

        m_algoName = repEntry.gtAlgoName();

        m_prescaleFactor = repEntry.gtPrescaleFactor();

        m_triggerMask = repEntry.gtTriggerMask();

    }

    return *this;

}

// equal operator
bool L1GtTrigReportEntry::operator==(const L1GtTrigReportEntry& repEntry) const
{

    if (m_triggerMenuName != repEntry.gtTriggerMenuName()) {
        return false;
    }

    if (m_algoName != repEntry.gtAlgoName()) {
        return false;
    }

    if (m_prescaleFactor != repEntry.gtPrescaleFactor()) {
        return false;
    }

    if (m_triggerMask != repEntry.gtTriggerMask()) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GtTrigReportEntry::operator!=(const L1GtTrigReportEntry& result) const
{

    return !( result == *this);

}

// member functions

/// increase # of events accepted/rejected for this entry
void L1GtTrigReportEntry::addValidEntry(const bool algResult) {

    if (algResult) {
        m_nrEventsAccept++;
    }
    else {
        m_nrEventsReject++;
    }
}

/// increase # of events with error 
void L1GtTrigReportEntry::addErrorEntry() {

    m_nrEventsError++;
}
