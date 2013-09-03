#ifndef L1GlobalTrigger_L1GlobalTriggerRecord_h
#define L1GlobalTrigger_L1GlobalTriggerRecord_h

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

// system include files
#include <vector>
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations
namespace edm
{
template <typename T> class Handle;
}

// class interface

class L1GlobalTriggerRecord
{

public:

    /// constructors
    L1GlobalTriggerRecord();

    L1GlobalTriggerRecord(const unsigned int numberPhysTriggers,
            const unsigned int numberTechnicalTriggers);

    /// copy constructor
    L1GlobalTriggerRecord(const L1GlobalTriggerRecord&);

    /// destructor
    virtual ~L1GlobalTriggerRecord();

    /// assignment operator
    L1GlobalTriggerRecord& operator=(const L1GlobalTriggerRecord&);

    /// equal operator
    bool operator==(const L1GlobalTriggerRecord&) const;

    /// unequal operator
    bool operator!=(const L1GlobalTriggerRecord&) const;

public:

    /// get Global Trigger decision, decision word and technical trigger word
    /// for bunch cross with L1Accept (BxInEvent = 0) 
    inline const bool decision() const {
        return m_gtGlobalDecision;
    }

    inline const DecisionWord decisionWord() const {
        return m_gtDecisionWord;
    }

    inline const TechnicalTriggerWord technicalTriggerWord() const {
        return m_gtTechnicalTriggerWord;
    }

    inline const DecisionWord decisionWordBeforeMask() const {
        return m_gtDecisionWordBeforeMask;
    }

    inline const TechnicalTriggerWord technicalTriggerWordBeforeMask() const {
        return m_gtTechnicalTriggerWordBeforeMask;
    }

    /// set global decision, decision word and technical trigger word
    /// for bunch cross with L1Accept (BxInEvent = 0) 
    void setDecision(const bool& dValue);
    void setDecisionWord(const DecisionWord& dWordValue);
    void setTechnicalTriggerWord(const TechnicalTriggerWord& ttWordValue);

    void setDecisionWordBeforeMask(const DecisionWord& dWordValue);
    void setTechnicalTriggerWordBeforeMask(
            const TechnicalTriggerWord& ttWordValue);

    /// get/set index of the set of prescale factors

    inline const unsigned int gtPrescaleFactorIndexTech() const {
        return m_gtPrescaleFactorIndexTech;
    }

    void setGtPrescaleFactorIndexTech(
            const unsigned int& gtPrescaleFactorIndexTechValue) {
        m_gtPrescaleFactorIndexTech = gtPrescaleFactorIndexTechValue;
    }

    inline const unsigned int gtPrescaleFactorIndexAlgo() const {
        return m_gtPrescaleFactorIndexAlgo;
    }

    void setGtPrescaleFactorIndexAlgo(
            const unsigned int& gtPrescaleFactorIndexAlgoValue) {
        m_gtPrescaleFactorIndexAlgo = gtPrescaleFactorIndexAlgoValue;
    }

    /// print global decision and algorithm decision word
    void printGtDecision(std::ostream& myCout) const;

    /// print technical triggers
    void printTechnicalTrigger(std::ostream& myCout) const;

    // other methods

    /// clear the record
    void reset();

    /// pretty print the content of a L1GlobalTriggerRecord
    void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream
            & operator<<(std::ostream&, const L1GlobalTriggerRecord&);

private:

    /// global decision for L1A bunch cross
    bool m_gtGlobalDecision;

    /// algorithm decision word for L1A bunch cross
    DecisionWord m_gtDecisionWord;

    /// technical trigger word for L1A bunch cross
    TechnicalTriggerWord m_gtTechnicalTriggerWord;

    /// algorithm decision word for L1A bunch cross before applying the masks
    DecisionWord m_gtDecisionWordBeforeMask;

    /// technical trigger word for L1A bunch cross before applying the masks
    TechnicalTriggerWord m_gtTechnicalTriggerWordBeforeMask;

    /// index of the set of prescale factors in the DB/EventSetup
    /// for algorithm triggers and technical triggers
    unsigned int m_gtPrescaleFactorIndexTech;
    unsigned int m_gtPrescaleFactorIndexAlgo;

};

#endif
