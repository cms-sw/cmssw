#ifndef GlobalTrigger_GtCaloCondition_h
#define GlobalTrigger_GtCaloCondition_h

/**
 * \class GtCaloCondition
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
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

// forward declarations
class GlobalCondition;
class GlobalCaloTemplate;

namespace l1t {

class L1Candidate;

class GtBoard;

// class declaration
class GtCaloCondition : public ConditionEvaluation
{

public:

    /// constructors
    ///     default
    GtCaloCondition();

    ///     from base template condition (from event setup usually)
    GtCaloCondition(const GlobalCondition*, const GtBoard*,
            const int nrL1EG,
            const int nrL1Jet,
            const int nrL1Tau,
            const int ifCaloEtaNumberBits);

    // copy constructor
    GtCaloCondition(const GtCaloCondition&);

    // destructor
    virtual ~GtCaloCondition();

    // assign operator
    GtCaloCondition& operator=(const GtCaloCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a Condition
    inline const GlobalCaloTemplate* gtCaloTemplate() const {
        return m_gtCaloTemplate;
    }

    void setGtCaloTemplate(const GlobalCaloTemplate*);

    ///   get / set the pointer to uGt GtBoard
    inline const GtBoard* getuGtB() const {
        return m_uGtB;
    }

    void setuGtB(const GtBoard*);


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
    void copy(const GtCaloCondition& cp);

    /// load calo candidates
    const l1t::L1Candidate* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool
    checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const;

private:

    /// pointer to a GlobalCaloTemplate
    const GlobalCaloTemplate* m_gtCaloTemplate;

    /// pointer to uGt GtBoard, to be able to get the trigger objects
    const GtBoard* m_uGtB;

    /// number of bits for eta of calorimeter objects
    int m_ifCaloEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;

};

}
#endif
