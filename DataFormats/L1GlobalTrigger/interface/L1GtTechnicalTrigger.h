#ifndef L1GlobalTrigger_L1GtTechnicalTrigger_h
#define L1GlobalTrigger_L1GtTechnicalTrigger_h

/**
 * \class L1GtTechnicalTrigger
 * 
 * 
 * Description: technical trigger input for L1 Global Trigger.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 * 
 *
 */

// system include files
#include <string>
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations
namespace edm {
  template <typename T>
  class Handle;
}

// class interface

class L1GtTechnicalTrigger {
public:
  /// constructors
  L1GtTechnicalTrigger();

  L1GtTechnicalTrigger(const std::string& ttName,
                       const unsigned int ttBitNumber,
                       const int ttBxInEvent,
                       const bool ttResult);

  /// copy constructor
  L1GtTechnicalTrigger(const L1GtTechnicalTrigger&);

  /// destructor
  virtual ~L1GtTechnicalTrigger();

  /// assignment operator
  L1GtTechnicalTrigger& operator=(const L1GtTechnicalTrigger&);

  /// equal operator
  bool operator==(const L1GtTechnicalTrigger&) const;

  /// unequal operator
  bool operator!=(const L1GtTechnicalTrigger&) const;

public:
  /// get / set technical trigger name, bit number, bunch cross in the GT event record and result

  inline const std::string gtTechnicalTriggerName() const { return m_gtTechnicalTriggerName; }

  inline const unsigned int gtTechnicalTriggerBitNumber() const { return m_gtTechnicalTriggerBitNumber; }

  /// get/set bunch cross in the GT event record
  inline const int bxInEvent() const { return m_bxInEvent; }

  inline const bool gtTechnicalTriggerResult() const { return m_gtTechnicalTriggerResult; }

  void setGtTechnicalTriggerName(const std::string& ttName);
  void setGtTechnicalTriggerBitNumber(const unsigned int ttBitNumber);
  void setBxInEvent(const int bxInEventValue);
  void setGtTechnicalTriggerResult(const bool ttResult);

  // other methods

  /// pretty print the content of a L1GtTechnicalTrigger
  void print(std::ostream& myCout) const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtTechnicalTrigger&);

private:
  /// technical trigger name, bit number, bunch cross in the GT event record and result
  std::string m_gtTechnicalTriggerName;
  unsigned int m_gtTechnicalTriggerBitNumber;
  int m_bxInEvent;
  bool m_gtTechnicalTriggerResult;
};

#endif
