#ifndef L1Trigger_L1TGlobal_CorrThreeBodyCondition_h
#define L1Trigger_L1TGlobal_CorrThreeBodyCondition_h

/**
 * \class CorrThreeBodyCondition
 *
 * Description: L1 Global Trigger three-body correlation conditions:                                                                      
 *              evaluation of a three-body correlation condition (= three-muon invariant mass)    
 *                                                                                                                                               
 * Implementation:                                                                                                                                             
 *    <TODO: enter implementation details>                                                                                                 
 *                                                                                                                                                                         
 * \author: Elisa Fontanesi - Boston University                                                                                                                                                            
 *          CorrCondition and CorrWithOverlapRemovalCondition classes used as a starting point
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
class CorrelationThreeBodyTemplate;

namespace l1t {

  class L1Candidate;

  class GlobalBoard;

  // class declaration
  class CorrThreeBodyCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    CorrThreeBodyCondition();

    ///     from base template condition (from event setup usually)
    CorrThreeBodyCondition(const GlobalCondition*,
                           const GlobalCondition*,
                           const GlobalCondition*,
                           const GlobalCondition*,
                           const GlobalBoard*

    );

    // copy constructor
    CorrThreeBodyCondition(const CorrThreeBodyCondition&);

    // destructor
    ~CorrThreeBodyCondition() override;

    // assign operator
    CorrThreeBodyCondition& operator=(const CorrThreeBodyCondition&);

  public:
    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

  public:
    ///   get / set the pointer to a Condition
    inline const CorrelationThreeBodyTemplate* gtCorrelationThreeBodyTemplate() const {
      return m_gtCorrelationThreeBodyTemplate;
    }

    void setGtCorrelationThreeBodyTemplate(const CorrelationThreeBodyTemplate*);

    ///   get / set the pointer to uGt GlobalBoard
    inline const GlobalBoard* getuGtB() const { return m_uGtB; }

    void setuGtB(const GlobalBoard*);

    void setScales(const GlobalScales*);

  private:
    ///  copy function for copy constructor and operator=
    void copy(const CorrThreeBodyCondition& cp);

    /// load  candidates
    const l1t::L1Candidate* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const;

  private:
    /// pointer to a CorrelationThreeBodyTemplate
    const CorrelationThreeBodyTemplate* m_gtCorrelationThreeBodyTemplate;

    // pointer to subconditions
    const GlobalCondition* m_gtCond0;
    const GlobalCondition* m_gtCond1;
    const GlobalCondition* m_gtCond2;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_uGtB;

    const GlobalScales* m_gtScales;
  };

}  // namespace l1t
#endif
