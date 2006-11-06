#ifndef L1GlobalTrigger_L1GlobalTriggerEvmReadoutRecord_h
#define L1GlobalTrigger_L1GlobalTriggerEvmReadoutRecord_h

/**
 * \class L1GlobalTriggerEvmReadoutRecord 
 * 
 * 
 * Description: EVM readout record for L1 Global Trigger 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

// forward declarations

// class interface

class L1GlobalTriggerEvmReadoutRecord
{

public:

    /// constructors
    L1GlobalTriggerEvmReadoutRecord();

    L1GlobalTriggerEvmReadoutRecord(int NumberBxInEvent);

    /// copy constructor
    L1GlobalTriggerEvmReadoutRecord(const L1GlobalTriggerEvmReadoutRecord&);

    /// destructor
    virtual ~L1GlobalTriggerEvmReadoutRecord();
      
    /// assignment operator
    L1GlobalTriggerEvmReadoutRecord& operator=(const L1GlobalTriggerEvmReadoutRecord&);
  
    /// equal operator
    bool operator==(const L1GlobalTriggerEvmReadoutRecord&) const;

    /// unequal operator
    bool operator!=(const L1GlobalTriggerEvmReadoutRecord&) const;

public:

    /// typedefs taken from L1GlobalTriggerReadoutSetup.h         

    /// algorithm bits: 128 bits
    typedef L1GlobalTriggerReadoutSetup::DecisionWord DecisionWord;

    /// extend DecisionWord with 64 bits (128 - 192)
    /// need a new FDL chip :-)  
    typedef L1GlobalTriggerReadoutSetup::DecisionWordExtended DecisionWordExtended;

    /// technical trigger bits (64 bits)
    typedef L1GlobalTriggerReadoutSetup::TechnicalTriggerWord TechnicalTriggerWord;

public:

    /// get Global Trigger decision and the decision word
    ///   overloaded w.r.t. bxInEvent argument
    ///   bxInEvent not given: for bunch cross with L1Accept
    const bool decision(unsigned int bxInEvent) const;
    const DecisionWord decisionWord(unsigned int bxInEvent) const;

    const bool decision() const;
    const DecisionWord decisionWord() const;
  
    /// set global decision and the decision word
    void setDecision(bool t, unsigned int bxInEvent);
    void setDecisionWord(const DecisionWord& decisionWordValue, unsigned int bxInEvent);

    void setDecision(bool t);
    void setDecisionWord(const DecisionWord& decisionWordValue);
    
    /// print global decision and algorithm decision word
    void printGtDecision(std::ostream& myCout, unsigned int bxInEventValue) const;
    void printGtDecision(std::ostream& myCout) const;
    
    /// print technical triggers
    void printTechnicalTrigger(std::ostream& myCout, unsigned int bxInEventValue) const;
    void printTechnicalTrigger(std::ostream& myCout) const;
    
    //**************************************************************************
    // get/set hardware-related words
    // 
    // Board description: file GlobalTriggerBoardsMapper.dat // TODO xml file instead?
    //**************************************************************************


    /// get / set GTFE word (record) in the GT readout record
    const L1GtfeWord gtfeWord() const;
    void setGtfeWord(const L1GtfeWord&);

    /// get / set TCS word (record) in the GT readout record
    const L1TcsWord tcsWord() const;
    void setTcsWord(const L1TcsWord&);

    /// get / set FDL word (record) in the GT readout record
    const L1GtFdlWord gtFdlWord(unsigned int bxInEvent) const;
    const L1GtFdlWord gtFdlWord() const;

    void setGtFdlWord(const L1GtFdlWord&, unsigned int bxInEvent);
    void setGtFdlWord(const L1GtFdlWord&);

    // other methods
    
    /// clear the record
    void reset(); 
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GlobalTriggerEvmReadoutRecord&);


private:

    L1GtfeWord m_gtfeWord;
    L1TcsWord m_tcsWord;
    
    std::vector<L1GtFdlWord> m_gtFdlWord;
      
};


#endif
