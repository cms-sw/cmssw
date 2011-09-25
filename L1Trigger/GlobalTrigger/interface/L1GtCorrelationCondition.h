#ifndef GlobalTrigger_L1GtCorrelationCondition_h
#define GlobalTrigger_L1GtCorrelationCondition_h

/**
 * \class L1GtCorrelationCondition
 *
 *
 * Description: evaluation of a CondCorrelation condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// forward declarations
class L1GtCondition;
class L1GtCorrelationTemplate;
class L1GlobalTriggerGTL;
class L1GlobalTriggerPSB;
class L1GtEtaPhiConversions;

// class declaration
class L1GtCorrelationCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtCorrelationCondition();

    ///     from base template condition (from event setup usually)
    L1GtCorrelationCondition(const L1GtCondition*, const L1GtCondition*,
            const L1GtCondition*, const int, const int, const int, const int,
            const L1GlobalTriggerGTL*, const L1GlobalTriggerPSB*,
            const L1GtEtaPhiConversions*);

    // copy constructor
    L1GtCorrelationCondition(const L1GtCorrelationCondition&);

    // destructor
    virtual ~L1GtCorrelationCondition();

    // assign operator
    L1GtCorrelationCondition& operator=(const L1GtCorrelationCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition() const;

    /// print condition
    void print(std::ostream& myCout) const;

public:

    ///   get / set the number of phi bins
    inline const unsigned int gtNrBinsPhi() const {
        return m_nrBinsPhi;
    }

    void setGtNrBinsPhi(const unsigned int);

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtCorrelationTemplate* gtCorrelationTemplate() const {
        return m_gtCorrelationTemplate;
    }

    void setGtCorrelationTemplate(const L1GtCorrelationTemplate*);

    ///   get / set the pointer to GTL
    inline const L1GlobalTriggerGTL* gtGTL() const {
        return m_gtGTL;
    }

    void setGtGTL(const L1GlobalTriggerGTL*);

    ///   get / set the pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const {
        return m_gtPSB;
    }

    void setGtPSB(const L1GlobalTriggerPSB*);

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtCorrelationCondition& cp);

private:

    /// pointer to a L1GtCorrelationTemplate
    const L1GtCorrelationTemplate* m_gtCorrelationTemplate;

    /// pointer to first sub-condition
    const L1GtCondition* m_gtCond0;

    /// pointer to second sub-condition
    const L1GtCondition* m_gtCond1;

    ///
    int m_cond0NrL1Objects;
    int m_cond1NrL1Objects;
    int m_cond0EtaBits;
    int m_cond1EtaBits;

    /// number of bins for delta phi
    unsigned int m_nrBinsPhi;

    /// pointer to GTL, to be able to get the trigger objects
    const L1GlobalTriggerGTL* m_gtGTL;

    /// pointer to PSB, to be able to get the trigger objects
    const L1GlobalTriggerPSB* m_gtPSB;

    /// pointer to eta and phi conversion class
    const L1GtEtaPhiConversions* m_gtEtaPhiConversions;

private:

    bool m_isDebugEnabled;


};

#endif
