#ifndef L1Trigger_L1TGlobal_EnergySumZdcCondition_h
#define L1Trigger_L1TGlobal_EnergySumZdcCondition_h

/**
 * \class EnergySumZdcCondition
 * 
 * 
 * Description: evaluation of a CondEnergySumZdc condition.
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
class EnergySumZdcTemplate;

namespace l1t {

  class L1Candidate;

  class GlobalBoard;

  // class declaration
  class EnergySumZdcCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    EnergySumZdcCondition();

    ///     from base template condition (from event setup usually)
    EnergySumZdcCondition(const GlobalCondition*, const GlobalBoard*);

    // copy constructor
    EnergySumZdcCondition(const EnergySumZdcCondition&);

    // destructor
    ~EnergySumZdcCondition() override;

    // assign operator
    EnergySumZdcCondition& operator=(const EnergySumZdcCondition&);

  public:
    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

  public:
    ///   get / set the pointer to a L1GtCondition
    inline const EnergySumZdcTemplate* gtEnergySumZdcTemplate() const { return m_gtEnergySumZdcTemplate; }

    void setGtEnergySumZdcTemplate(const EnergySumZdcTemplate*);

    ///   get / set the pointer to uGt GlobalBoard
    inline const GlobalBoard* getuGtB() const { return m_uGtB; }

    void setuGtB(const GlobalBoard*);

  private:
    /// copy function for copy constructor and operator=
    void copy(const EnergySumZdcCondition& cp);

  private:
    /// pointer to a EnergySumZdcTemplate
    const EnergySumZdcTemplate* m_gtEnergySumZdcTemplate;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_uGtB;
  };

}  // namespace l1t
#endif
