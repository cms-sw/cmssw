#ifndef L1Trigger_L1TGlobal_CaloCondition_h
#define L1Trigger_L1TGlobal_CaloCondition_h

/**
 * \class CaloCondition
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
class CaloTemplate;

namespace l1t {

class L1Candidate;

class GlobalBoard;

// class declaration
class CaloCondition : public ConditionEvaluation
{

public:

    /// constructors
    ///     default
    CaloCondition();

    ///     from base template condition (from event setup usually)
    CaloCondition(const GlobalCondition*, const GlobalBoard*,
            const int nrL1EG,
            const int nrL1Jet,
            const int nrL1Tau,
            const int ifCaloEtaNumberBits);

    // copy constructor
    CaloCondition(const CaloCondition&);

    // destructor
    virtual ~CaloCondition();

    // assign operator
    CaloCondition& operator=(const CaloCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a Condition
    inline const CaloTemplate* gtCaloTemplate() const {
        return m_gtCaloTemplate;
    }

    void setGtCaloTemplate(const CaloTemplate*);

    ///   get / set the pointer to uGt GlobalBoard
    inline const GlobalBoard* getuGtB() const {
        return m_uGtB;
    }

    void setuGtB(const GlobalBoard*);


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
    void copy(const CaloCondition& cp);

    /// load calo candidates
    const l1t::L1Candidate* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool
    checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const;

private:

    /// pointer to a CaloTemplate
    const CaloTemplate* m_gtCaloTemplate;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_uGtB;

    /// number of bits for eta of calorimeter objects
    int m_ifCaloEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;

};

}
#endif
