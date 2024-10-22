/**
 * \class L1GlobalTriggerRecord
 * 
 * 
 * Description: stripped-down record for L1 Global Trigger.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 * 
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"

// system include files
#include <iomanip>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GlobalTriggerRecord::L1GlobalTriggerRecord() {
  // empty
}

L1GlobalTriggerRecord::L1GlobalTriggerRecord(const unsigned int numberPhysTriggers,
                                             const unsigned int numberTechnicalTriggers) {
  m_gtDecisionWord.reserve(numberPhysTriggers);
  m_gtTechnicalTriggerWord.reserve(numberTechnicalTriggers);

  m_gtDecisionWordBeforeMask.reserve(numberPhysTriggers);
  m_gtTechnicalTriggerWordBeforeMask.reserve(numberTechnicalTriggers);
}

// copy constructor
L1GlobalTriggerRecord::L1GlobalTriggerRecord(const L1GlobalTriggerRecord& result) {
  m_gtGlobalDecision = result.m_gtGlobalDecision;

  m_gtDecisionWord = result.m_gtDecisionWord;
  m_gtTechnicalTriggerWord = result.m_gtTechnicalTriggerWord;

  m_gtDecisionWordBeforeMask = result.m_gtDecisionWordBeforeMask;
  m_gtTechnicalTriggerWordBeforeMask = result.m_gtTechnicalTriggerWordBeforeMask;

  m_gtPrescaleFactorIndexTech = result.m_gtPrescaleFactorIndexTech;
  m_gtPrescaleFactorIndexAlgo = result.m_gtPrescaleFactorIndexAlgo;
}

// destructor
L1GlobalTriggerRecord::~L1GlobalTriggerRecord() {
  // empty now
}

// assignment operator
L1GlobalTriggerRecord& L1GlobalTriggerRecord::operator=(const L1GlobalTriggerRecord& result) {
  if (this != &result) {
    m_gtGlobalDecision = result.m_gtGlobalDecision;

    m_gtDecisionWord = result.m_gtDecisionWord;
    m_gtTechnicalTriggerWord = result.m_gtTechnicalTriggerWord;

    m_gtDecisionWordBeforeMask = result.m_gtDecisionWordBeforeMask;
    m_gtTechnicalTriggerWordBeforeMask = result.m_gtTechnicalTriggerWordBeforeMask;

    m_gtPrescaleFactorIndexTech = result.m_gtPrescaleFactorIndexTech;
    m_gtPrescaleFactorIndexAlgo = result.m_gtPrescaleFactorIndexAlgo;
  }

  return *this;
}

// equal operator
bool L1GlobalTriggerRecord::operator==(const L1GlobalTriggerRecord& result) const {
  if (m_gtGlobalDecision != result.m_gtGlobalDecision) {
    return false;
  }

  if (m_gtDecisionWord != result.m_gtDecisionWord) {
    return false;
  }

  if (m_gtTechnicalTriggerWord != result.m_gtTechnicalTriggerWord) {
    return false;
  }

  if (m_gtDecisionWordBeforeMask != result.m_gtDecisionWordBeforeMask) {
    return false;
  }

  if (m_gtTechnicalTriggerWordBeforeMask != result.m_gtTechnicalTriggerWordBeforeMask) {
    return false;
  }

  if (m_gtPrescaleFactorIndexTech != result.m_gtPrescaleFactorIndexTech) {
    return false;
  }

  if (m_gtPrescaleFactorIndexAlgo != result.m_gtPrescaleFactorIndexAlgo) {
    return false;
  }

  // all members identical
  return true;
}

// unequal operator
bool L1GlobalTriggerRecord::operator!=(const L1GlobalTriggerRecord& result) const { return !(result == *this); }
// methods

// set global decision
void L1GlobalTriggerRecord::setDecision(const bool& dValue) { m_gtGlobalDecision = dValue; }

// set decision word
void L1GlobalTriggerRecord::setDecisionWord(const DecisionWord& dWordValue) { m_gtDecisionWord = dWordValue; }

void L1GlobalTriggerRecord::setTechnicalTriggerWord(const TechnicalTriggerWord& ttWordValue) {
  m_gtTechnicalTriggerWord = ttWordValue;
}

// set decision word before applying the masks
void L1GlobalTriggerRecord::setDecisionWordBeforeMask(const DecisionWord& dWordValue) {
  m_gtDecisionWordBeforeMask = dWordValue;
}

void L1GlobalTriggerRecord::setTechnicalTriggerWordBeforeMask(const TechnicalTriggerWord& ttWordValue) {
  m_gtTechnicalTriggerWordBeforeMask = ttWordValue;
}

// print global decision and algorithm decision word
void L1GlobalTriggerRecord::printGtDecision(std::ostream& myCout) const {
  myCout << std::endl;
  myCout << "\n  Global decision (FinalOR): " << m_gtGlobalDecision << std::endl;

  // decision word (in 64bits words)
  int sizeW64 = 64;  // 64 bits words

  int iBit = 0;
  int nrDecWord = m_gtDecisionWord.size() / sizeW64;

  std::ostringstream stream64;

  std::vector<std::string> decWord;
  decWord.reserve(nrDecWord);

  for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecisionWord.rbegin(); ritBit != m_gtDecisionWord.rend();
       ++ritBit) {
    stream64 << (*ritBit ? '1' : '0');

    if ((((iBit + 1) % 16) == (sizeW64 % 16))) {
      stream64 << " ";
    }

    if (((iBit + 1) % sizeW64) == 0) {
      std::string iW = stream64.str();
      stream64.str("");

      decWord.push_back(iW);
    }

    iBit++;
  }

  int iWord = 0;

  myCout << "\n  DecisionWord after trigger mask (bitset style)";
  for (std::vector<std::string>::reverse_iterator ritWord = decWord.rbegin(); ritWord != decWord.rend(); ++ritWord) {
    myCout << "\n  Bits " << iWord * sizeW64 + sizeW64 - 1 << " : " << iWord * sizeW64 << "\n  ";
    myCout << *ritWord;

    iWord++;
  }
}

// print technical trigger word (reverse order for vector<bool>)
void L1GlobalTriggerRecord::printTechnicalTrigger(std::ostream& myCout) const {
  myCout << "\n  Technical triggers after trigger mask  (bitset style):    \n  ";

  int sizeW64 = 64;  // 64 bits words
  int iBit = 0;

  for (std::vector<bool>::const_reverse_iterator ritBit = m_gtTechnicalTriggerWord.rbegin();
       ritBit != m_gtTechnicalTriggerWord.rend();
       ++ritBit) {
    myCout << (*ritBit ? '1' : '0');

    if ((((iBit + 1) % 16) == (sizeW64 % 16)) && (iBit != 63)) {
      myCout << " ";
    }

    iBit++;
  }
}

// clear the record
void L1GlobalTriggerRecord::reset() {
  m_gtGlobalDecision = false;

  for (std::vector<bool>::iterator itBit = m_gtDecisionWord.begin(); itBit != m_gtDecisionWord.end(); ++itBit) {
    *itBit = false;
  }
  for (std::vector<bool>::iterator itBit = m_gtTechnicalTriggerWord.begin(); itBit != m_gtTechnicalTriggerWord.end();
       ++itBit) {
    *itBit = false;
  }
  for (std::vector<bool>::iterator itBit = m_gtDecisionWordBeforeMask.begin();
       itBit != m_gtDecisionWordBeforeMask.end();
       ++itBit) {
    *itBit = false;
  }

  for (std::vector<bool>::iterator itBit = m_gtTechnicalTriggerWordBeforeMask.begin();
       itBit != m_gtTechnicalTriggerWordBeforeMask.end();
       ++itBit) {
    *itBit = false;
  }

  m_gtPrescaleFactorIndexTech = 0;
  m_gtPrescaleFactorIndexAlgo = 0;
}

// pretty print the content of a L1GlobalTriggerRecord
void L1GlobalTriggerRecord::print(std::ostream& myCout) const {
  printGtDecision(myCout);

  myCout << std::endl;

  printTechnicalTrigger(myCout);

  myCout << std::endl;
  myCout << "\n  Index of prescale factor set for technical triggers: " << m_gtPrescaleFactorIndexTech << std::endl;
  myCout << "  Index of prescale factor set for algorithm triggers: " << m_gtPrescaleFactorIndexTech << std::endl;
  myCout << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& streamRec, const L1GlobalTriggerRecord& result) {
  result.print(streamRec);
  return streamRec;
}
