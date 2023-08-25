#ifndef L1Trigger_L1TGlobal_ZdcEnergySumCondition_h
#define L1Trigger_L1TGlobal_ZdcEnergySumCondition_h

/**
 * \class ZdcEnergySumCondition
 * 
 * 
 * Description: evaluation of a CondZdcEnergySum condition.
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
class ZdcEnergySumTemplate;

namespace l1t {

  class L1Candidate;

  class GlobalBoard;

  // class declaration
  class ZdcEnergySumCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    ZdcEnergySumCondition();

    ///     from base template condition (from event setup usually)
    ZdcEnergySumCondition(const GlobalCondition*, const GlobalBoard*);

    // copy constructor
    ZdcEnergySumCondition(const ZdcEnergySumCondition&);

    // destructor
    ~ZdcEnergySumCondition() override;

    // assign operator
    ZdcEnergySumCondition& operator=(const ZdcEnergySumCondition&);

  public:
    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

  public:
    ///   get / set the pointer to a L1GtCondition
    inline const ZdcEnergySumTemplate* gtZdcEnergySumTemplate() const { return m_gtZdcEnergySumTemplate; }

    void setGtZdcEnergySumTemplate(const ZdcEnergySumTemplate*);

    ///   get / set the pointer to uGt GlobalBoard
    inline const GlobalBoard* getuGtB() const { return m_uGtB; }

    void setuGtB(const GlobalBoard*);

  private:
    /// copy function for copy constructor and operator=
    void copy(const ZdcEnergySumCondition& cp);

  private:
    /// pointer to a ZdcEnergySumTemplate
    const ZdcEnergySumTemplate* m_gtZdcEnergySumTemplate;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_uGtB;
  };

}  // namespace l1t
#endif
