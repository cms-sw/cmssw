#ifndef L1GlobalTrigger_L1GlobalTriggerReadoutRecord_h
#define L1GlobalTrigger_L1GlobalTriggerReadoutRecord_h

/**
 * \class L1GlobalTriggerReadoutRecord
 * 
 * 
 * Description: readout record for L1 Global Trigger.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: N. Neumeister        - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 *
 */

// system include files
#include <string>
#include <vector>
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "DataFormats/Common/interface/RefProd.h"

#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations
namespace edm {
  template <typename T>
  class Handle;
}

// class interface

class L1GlobalTriggerReadoutRecord {
public:
  /// constructors
  L1GlobalTriggerReadoutRecord();

  L1GlobalTriggerReadoutRecord(int NumberBxInEvent);

  L1GlobalTriggerReadoutRecord(const int numberBxInEvent, const int numberFdlBoards, const int numberPsbBoards);

  /// copy constructor
  L1GlobalTriggerReadoutRecord(const L1GlobalTriggerReadoutRecord&);

  /// destructor
  virtual ~L1GlobalTriggerReadoutRecord();

  /// assignment operator
  L1GlobalTriggerReadoutRecord& operator=(const L1GlobalTriggerReadoutRecord&);

  /// equal operator
  bool operator==(const L1GlobalTriggerReadoutRecord&) const;

  /// unequal operator
  bool operator!=(const L1GlobalTriggerReadoutRecord&) const;

public:
  /// get Global Trigger decision, decision word and technical trigger word
  ///   overloaded w.r.t. bxInEvent argument
  ///   bxInEvent not given: for bunch cross with L1Accept
  const bool decision(int bxInEventValue) const;
  const bool decision() const;

  const cms_uint16_t finalOR(int bxInEventValue) const;
  const cms_uint16_t finalOR() const;

  const DecisionWord& decisionWord(int bxInEventValue) const;
  const DecisionWord& decisionWord() const;

  const TechnicalTriggerWord& technicalTriggerWord(int bxInEventValue) const;
  const TechnicalTriggerWord& technicalTriggerWord() const;

  /// set global decision, decision word and technical trigger word
  void setDecision(const bool& t, int bxInEventValue);
  void setDecision(const bool& t);

  void setDecisionWord(const DecisionWord& decisionWordValue, int bxInEventValue);
  void setDecisionWord(const DecisionWord& decisionWordValue);

  void setTechnicalTriggerWord(const TechnicalTriggerWord& ttWordValue, int bxInEventValue);
  void setTechnicalTriggerWord(const TechnicalTriggerWord& ttWordValue);

  /// print global decision and algorithm decision word
  void printGtDecision(std::ostream& myCout, int bxInEventValue) const;
  void printGtDecision(std::ostream& myCout) const;

  /// print technical triggers
  void printTechnicalTrigger(std::ostream& myCout, int bxInEventValue) const;
  void printTechnicalTrigger(std::ostream& myCout) const;

  /// get / set reference to L1MuGMTReadoutCollection
  const edm::RefProd<L1MuGMTReadoutCollection> muCollectionRefProd() const;
  void setMuCollectionRefProd(edm::Handle<L1MuGMTReadoutCollection>&);
  void setMuCollectionRefProd(const edm::RefProd<L1MuGMTReadoutCollection>&);

  //**************************************************************************
  // get/set hardware-related words
  //
  //**************************************************************************

  /// get / set GTFE word (record) in the GT readout record
  const L1GtfeWord gtfeWord() const;
  void setGtfeWord(const L1GtfeWord&);

  /// get the vector of L1GtFdlWord
  const std::vector<L1GtFdlWord> gtFdlVector() const { return m_gtFdlWord; }

  std::vector<L1GtFdlWord>& gtFdlVector() { return m_gtFdlWord; }

  /// get / set FDL word (record) in the GT readout record
  const L1GtFdlWord gtFdlWord(int bxInEventValue) const;
  const L1GtFdlWord gtFdlWord() const;

  void setGtFdlWord(const L1GtFdlWord&, int bxInEventValue);
  void setGtFdlWord(const L1GtFdlWord&);

  /// get the vector of L1GtPsbWord
  const std::vector<L1GtPsbWord> gtPsbVector() const { return m_gtPsbWord; }

  std::vector<L1GtPsbWord>& gtPsbVector() { return m_gtPsbWord; }

  /// get / set PSB word (record) in the GT readout record
  const L1GtPsbWord gtPsbWord(cms_uint16_t boardIdValue, int bxInEventValue) const;
  const L1GtPsbWord gtPsbWord(cms_uint16_t boardIdValue) const;

  void setGtPsbWord(const L1GtPsbWord&, cms_uint16_t boardIdValue, int bxInEventValue);
  void setGtPsbWord(const L1GtPsbWord&, cms_uint16_t boardIdValue);
  void setGtPsbWord(const L1GtPsbWord& gtPsbWordValue);

  // other methods

  /// clear the record
  void reset();

  /// pretty print the content of a L1GlobalTriggerReadoutRecord
  void print(std::ostream& myCout) const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GlobalTriggerReadoutRecord&);

private:
  L1GtfeWord m_gtfeWord;

  std::vector<L1GtFdlWord> m_gtFdlWord;

  std::vector<L1GtPsbWord> m_gtPsbWord;

  edm::RefProd<L1MuGMTReadoutCollection> m_muCollRefProd;
};

#endif
