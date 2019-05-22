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
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"

// system include files
#include <string>

// user include files

// constructor(s)
L1GtTrigReportEntry::L1GtTrigReportEntry(const std::string& menuName,
                                         const std::string& algName,
                                         const int prescaleFactor,
                                         const int triggerMask,
                                         const int daqPartition) {
  m_triggerMenuName = menuName;
  m_algoName = algName;
  m_prescaleFactor = prescaleFactor;
  m_triggerMask = triggerMask;
  m_daqPartition = daqPartition;

  // counters

  m_nrEventsAccept = 0;
  m_nrEventsReject = 0;
  m_nrEventsAcceptBeforeMask = 0;
  m_nrEventsRejectBeforeMask = 0;
  m_nrEventsError = 0;
}

// destructor
L1GtTrigReportEntry::~L1GtTrigReportEntry() {
  //empty
}

// assignment operator
L1GtTrigReportEntry& L1GtTrigReportEntry::operator=(const L1GtTrigReportEntry& repEntry) {
  if (this != &repEntry) {
    m_triggerMenuName = repEntry.m_triggerMenuName;

    m_algoName = repEntry.m_algoName;

    m_prescaleFactor = repEntry.m_prescaleFactor;

    m_triggerMask = repEntry.m_triggerMask;

    m_daqPartition = repEntry.m_daqPartition;
  }

  return *this;
}

// equal operator
bool L1GtTrigReportEntry::operator==(const L1GtTrigReportEntry& repEntry) const {
  // faster comparisons first

  if (m_daqPartition != repEntry.m_daqPartition) {
    return false;
  }

  if (m_triggerMask != repEntry.m_triggerMask) {
    return false;
  }

  if (m_prescaleFactor != repEntry.m_prescaleFactor) {
    return false;
  }

  if (m_algoName != repEntry.m_algoName) {
    return false;
  }

  if (m_triggerMenuName != repEntry.m_triggerMenuName) {
    return false;
  }

  // all members identical
  return true;
}

// unequal operator
bool L1GtTrigReportEntry::operator!=(const L1GtTrigReportEntry& result) const { return !(result == *this); }

// member functions

/// increase # of events accepted/rejected for this entry
void L1GtTrigReportEntry::addValidEntry(const bool algResultAfterMask, const bool algResultBeforeMask) {
  if (algResultAfterMask) {
    m_nrEventsAccept++;
  } else {
    m_nrEventsReject++;
  }

  if (algResultBeforeMask) {
    m_nrEventsAcceptBeforeMask++;
  } else {
    m_nrEventsRejectBeforeMask++;
  }
}

/// increase # of events with error
void L1GtTrigReportEntry::addErrorEntry() { m_nrEventsError++; }
