/**
 * \class L1GtTechnicalTrigger
 *
 *
 * Description: technical trigger input record for L1 Global Trigger.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTrigger.h"

// system include files
#include <iomanip>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GtTechnicalTrigger::L1GtTechnicalTrigger()
    : m_gtTechnicalTriggerName(""),
      m_gtTechnicalTriggerBitNumber(0),
      m_bxInEvent(0),
      m_gtTechnicalTriggerResult(false) {
  // empty
}

L1GtTechnicalTrigger::L1GtTechnicalTrigger(const std::string& ttName,
                                           const unsigned int ttBitNumber,
                                           const int ttBxInEvent,
                                           const bool ttResult)
    : m_gtTechnicalTriggerName(ttName),
      m_gtTechnicalTriggerBitNumber(ttBitNumber),
      m_bxInEvent(ttBxInEvent),
      m_gtTechnicalTriggerResult(ttResult)

{
  // empty
}

// copy constructor
L1GtTechnicalTrigger::L1GtTechnicalTrigger(const L1GtTechnicalTrigger& result) {
  m_gtTechnicalTriggerName = result.m_gtTechnicalTriggerName;
  m_gtTechnicalTriggerBitNumber = result.m_gtTechnicalTriggerBitNumber;
  m_bxInEvent = result.m_bxInEvent;
  m_gtTechnicalTriggerResult = result.m_gtTechnicalTriggerResult;
}

// destructor
L1GtTechnicalTrigger::~L1GtTechnicalTrigger() {
  // empty now
}

// assignment operator
L1GtTechnicalTrigger& L1GtTechnicalTrigger::operator=(const L1GtTechnicalTrigger& result) {
  if (this != &result) {
    m_gtTechnicalTriggerName = result.m_gtTechnicalTriggerName;
    m_gtTechnicalTriggerBitNumber = result.m_gtTechnicalTriggerBitNumber;
    m_bxInEvent = result.m_bxInEvent;
    m_gtTechnicalTriggerResult = result.m_gtTechnicalTriggerResult;
  }

  return *this;
}

// equal operator
bool L1GtTechnicalTrigger::operator==(const L1GtTechnicalTrigger& result) const {
  if (m_gtTechnicalTriggerName != result.m_gtTechnicalTriggerName) {
    return false;
  }

  if (m_gtTechnicalTriggerBitNumber != result.m_gtTechnicalTriggerBitNumber) {
    return false;
  }

  if (m_bxInEvent != result.m_bxInEvent) {
    return false;
  }

  if (m_gtTechnicalTriggerResult != result.m_gtTechnicalTriggerResult) {
    return false;
  }

  // all members identical
  return true;
}

// unequal operator
bool L1GtTechnicalTrigger::operator!=(const L1GtTechnicalTrigger& result) const { return !(result == *this); }
// methods

// set technical trigger name, bit number and result

void L1GtTechnicalTrigger::setGtTechnicalTriggerName(const std::string& ttName) { m_gtTechnicalTriggerName = ttName; }

// set decision word
void L1GtTechnicalTrigger::setGtTechnicalTriggerBitNumber(const unsigned int ttBitNumber) {
  m_gtTechnicalTriggerBitNumber = ttBitNumber;
}

void L1GtTechnicalTrigger::setBxInEvent(const int bxInEventValue) { m_bxInEvent = bxInEventValue; }

void L1GtTechnicalTrigger::setGtTechnicalTriggerResult(const bool ttResult) { m_gtTechnicalTriggerResult = ttResult; }

// pretty print the content of a L1GtTechnicalTrigger
void L1GtTechnicalTrigger::print(std::ostream& myCout) const {
  myCout << std::endl;
  myCout << std::endl;
  myCout << "\nTechnical trigger name: " << m_gtTechnicalTriggerName << std::endl;
  myCout << "   bit number" << m_gtTechnicalTriggerBitNumber << std::endl;
  myCout << "   bxInEvent" << m_bxInEvent << std::endl;
  myCout << "   result" << m_gtTechnicalTriggerResult << std::endl;

  /// bunch cross in the GT event record
}

// output stream operator
std::ostream& operator<<(std::ostream& streamRec, const L1GtTechnicalTrigger& result) {
  result.print(streamRec);
  return streamRec;
}
