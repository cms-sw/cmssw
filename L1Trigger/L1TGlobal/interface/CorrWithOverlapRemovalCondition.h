#ifndef L1Trigger_L1TGlobal_CorrWithOverlapRemovalCondition_h
#define L1Trigger_L1TGlobal_CorrWithOverlapRemovalCondition_h

/**
 * \class CorrWithOverlapRemovalCondition
 * 
 * 
 * Description: evaluation of a correlation with overlap removal condition.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vladimir Rekovic
 *
 * 
 *
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"

// forward declarations
class GlobalCondition;
class CorrelationWithOverlapRemovalTemplate;

namespace l1t {

class L1Candidate;

class GlobalBoard;

// class declaration
class CorrWithOverlapRemovalCondition : public ConditionEvaluation
{

public:

    /// constructors
    ///     default
    CorrWithOverlapRemovalCondition();

    ///     from base template condition (from event setup usually)
    CorrWithOverlapRemovalCondition(const GlobalCondition*,
                  const GlobalCondition*,
		  const GlobalCondition*, 
		  const GlobalCondition*, 
                  const GlobalBoard*

            );

    // copy constructor
    CorrWithOverlapRemovalCondition(const CorrWithOverlapRemovalCondition&);

    // destructor
    virtual ~CorrWithOverlapRemovalCondition();

    // assign operator
    CorrWithOverlapRemovalCondition& operator=(const CorrWithOverlapRemovalCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a Condition
    inline const CorrelationWithOverlapRemovalTemplate* gtCorrelationWithOverlapRemovalTemplate() const {
        return m_gtCorrelationWithOverlapRemovalTemplate;
    }

    void setGtCorrelationWithOverlapRemovalTemplate(const CorrelationWithOverlapRemovalTemplate*);

    ///   get / set the pointer to uGt GlobalBoard
    inline const GlobalBoard* getuGtB() const {
        return m_uGtB;
    }

    void setuGtB(const GlobalBoard*);
    
    void setScales(const GlobalScales*);  

/*   //BLW Comment out for now
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
*/
private:

    ///  copy function for copy constructor and operator=
    void copy(const CorrWithOverlapRemovalCondition& cp);

    /// load  candidates
    const l1t::L1Candidate* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool
    checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const;

private:

    /// pointer to a CorrelationWithOverlapRemovalTemplate
    const CorrelationWithOverlapRemovalTemplate* m_gtCorrelationWithOverlapRemovalTemplate;


    // pointer to subconditions
    const GlobalCondition* m_gtCond0;
    const GlobalCondition* m_gtCond1;
    const GlobalCondition* m_gtCond2; // used for overlap removal

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_uGtB;
    
    const GlobalScales* m_gtScales;


/*   //BLW comment out for now
    /// number of bits for eta of calorimeter objects
    int m_ifCaloEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;
*/


};

}
#endif
