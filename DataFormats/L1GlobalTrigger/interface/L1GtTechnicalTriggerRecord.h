#ifndef L1GlobalTrigger_L1GtTechnicalTriggerRecord_h
#define L1GlobalTrigger_L1GtTechnicalTriggerRecord_h

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

// system include files
#include <string>
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTrigger.h"

// forward declarations

// class declaration
class L1GtTechnicalTriggerRecord {
public:
  /// constructor(s)
  L1GtTechnicalTriggerRecord();

  /// destructor
  virtual ~L1GtTechnicalTriggerRecord();

public:
  /// return the technical trigger for ttName and bxInEvent
  const L1GtTechnicalTrigger* getTechnicalTrigger(const std::string& ttName, const int bxInEventVal) const;

  /// return the technical trigger for ttBitNumber and bxInEvent
  const L1GtTechnicalTrigger* getTechnicalTrigger(const unsigned int ttBitNumber, const int bxInEventVal) const;

public:
  /// get / set the vector of technical triggers
  inline const std::vector<L1GtTechnicalTrigger>& gtTechnicalTrigger() const { return m_gtTechnicalTrigger; }

  void setGtTechnicalTrigger(const std::vector<L1GtTechnicalTrigger>& gtTechnicalTriggerValue) {
    m_gtTechnicalTrigger = gtTechnicalTriggerValue;
  }

private:
  std::vector<L1GtTechnicalTrigger> m_gtTechnicalTrigger;
};

#endif /* L1GlobalTrigger_L1GtTechnicalTriggerRecord_h */
