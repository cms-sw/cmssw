#ifndef L1GlobalTrigger_L1GlobalTriggerEvmReadoutRecord_h
#define L1GlobalTrigger_L1GlobalTriggerEvmReadoutRecord_h

/**
 * \class L1GlobalTriggerEvmReadoutRecord
 * 
 * 
 * Description: EVM readout record for L1 Global Trigger.  
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
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
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

    L1GlobalTriggerEvmReadoutRecord(const int numberBxInEvent, const int numberFdlBoards);

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

    /// get Global Trigger decision and the decision word
    ///   overloaded w.r.t. bxInEvent argument
    ///   bxInEvent not given: for bunch cross with L1Accept
    const bool decision(int bxInEvent) const;
    const DecisionWord decisionWord(int bxInEvent) const;

    const bool decision() const;
    const DecisionWord decisionWord() const;

    /// set global decision and the decision word
    void setDecision(bool t, int bxInEvent);
    void setDecisionWord(const DecisionWord& decisionWordValue, int bxInEvent);

    void setDecision(bool t);
    void setDecisionWord(const DecisionWord& decisionWordValue);

    /// print global decision and algorithm decision word
    void printGtDecision(std::ostream& myCout, int bxInEventValue) const;
    void printGtDecision(std::ostream& myCout) const;

    /// print technical triggers
    void printTechnicalTrigger(std::ostream& myCout, int bxInEventValue) const;
    void printTechnicalTrigger(std::ostream& myCout) const;

    //**************************************************************************
    // get/set hardware-related words
    //
    // Board description: file GlobalTriggerBoardsMapper.dat // TODO xml file instead?
    //**************************************************************************


    /// get / set GTFE word (record) in the GT readout record
    const L1GtfeExtWord gtfeWord() const;
    void setGtfeWord(const L1GtfeExtWord&);

    /// get / set TCS word (record) in the GT readout record
    const L1TcsWord tcsWord() const;
    void setTcsWord(const L1TcsWord&);

    /// get the vector of L1GtFdlWord
    const std::vector<L1GtFdlWord> gtFdlVector() const
    {
        return m_gtFdlWord;
    }

    std::vector<L1GtFdlWord>& gtFdlVector()
    {
        return m_gtFdlWord;
    }

    /// get / set FDL word (record) in the GT readout record
    const L1GtFdlWord gtFdlWord(int bxInEvent) const;
    const L1GtFdlWord gtFdlWord() const;

    void setGtFdlWord(const L1GtFdlWord&, int bxInEvent);
    void setGtFdlWord(const L1GtFdlWord&);

    // other methods

    /// clear the record
    void reset();

    /// pretty print the content of a L1GlobalTriggerEvmReadoutRecord
    void print(std::ostream& myCout) const;


    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GlobalTriggerEvmReadoutRecord&);


private:

    L1GtfeExtWord m_gtfeWord;
    L1TcsWord m_tcsWord;

    std::vector<L1GtFdlWord> m_gtFdlWord;

};


#endif
