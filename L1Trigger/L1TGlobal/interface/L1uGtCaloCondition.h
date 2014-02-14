#ifndef GlobalTrigger_L1uGtCaloCondition_h
#define GlobalTrigger_L1uGtCaloCondition_h

/**
 * \class L1uGtCaloCondition
 * 
 * 
 * Description: evaluation of a CondCalo condition.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 *
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/L1uGtConditionEvaluation.h"

// forward declarations
class L1uGtCondition;
class L1uGtCaloTemplate;

namespace l1t {

class L1Candidate;

class L1uGtBoard;

// class declaration
class L1uGtCaloCondition : public L1uGtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1uGtCaloCondition();

    ///     from base template condition (from event setup usually)
    L1uGtCaloCondition(const L1uGtCondition*, const L1uGtBoard*,
            const int nrL1EG,
            const int nrL1Jet,
            const int nrL1Tau,
            const int ifCaloEtaNumberBits);

    // copy constructor
    L1uGtCaloCondition(const L1uGtCaloCondition&);

    // destructor
    virtual ~L1uGtCaloCondition();

    // assign operator
    L1uGtCaloCondition& operator=(const L1uGtCaloCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1uGtCondition
    inline const L1uGtCaloTemplate* gtCaloTemplate() const {
        return m_gtCaloTemplate;
    }

    void setGtCaloTemplate(const L1uGtCaloTemplate*);

    ///   get / set the pointer to uGt Board
    inline const L1uGtBoard* getuGtB() const {
        return m_uGtB;
    }

    void setuGtB(const L1uGtBoard*);


    ///   get / set the number of bits for eta of calorimeter objects
    inline const int gtIfCaloEtaNumberBits() const {
        return m_ifCaloEtaNumberBits;
    }

    void setGtIfCaloEtaNumberBits(const int&);

    ///   get / set maximum number of bins for the delta phi scales
    inline const int gtCorrParDeltaPhiNrBins() const {
        return m_corrParDeltaPhiNrBins;
    }

    void setGtCorrParDeltaPhiNrBins(const int&);

private:

    ///  copy function for copy constructor and operator=
    void copy(const L1uGtCaloCondition& cp);

    /// load calo candidates
    const l1t::L1Candidate* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool
    checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const;

private:

    /// pointer to a L1uGtCaloTemplate
    const L1uGtCaloTemplate* m_gtCaloTemplate;

    /// pointer to uGt Board, to be able to get the trigger objects
    const L1uGtBoard* m_uGtB;

    /// number of bits for eta of calorimeter objects
    int m_ifCaloEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;

};

}
#endif
