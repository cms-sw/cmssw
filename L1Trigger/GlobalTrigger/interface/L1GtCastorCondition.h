#ifndef GlobalTrigger_L1GtCastorCondition_h
#define GlobalTrigger_L1GtCastorCondition_h

/**
 * \class L1GtCastorCondition
 *
 *
 * Description: evaluation of a CondCastor condition.
 *
 * Implementation:
 *    Simply put the result read from CASTOR L1 record in the
 * L1GtConditionEvaluation base class, to be similar with other conditions.
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
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// forward declarations
class L1GtCondition;
class L1GtCastorTemplate;

// class declaration
class L1GtCastorCondition : public L1GtConditionEvaluation {
public:
  /// constructors
  ///     default
  L1GtCastorCondition();

  ///     from base template condition (from event setup usually)
  L1GtCastorCondition(const L1GtCondition *, const bool result);

  // copy constructor
  L1GtCastorCondition(const L1GtCastorCondition &);

  // destructor
  ~L1GtCastorCondition() override;

  // assign operator
  L1GtCastorCondition &operator=(const L1GtCastorCondition &);

public:
  /// the core function to check if the condition matches
  const bool evaluateCondition() const override;

  /// print condition
  void print(std::ostream &myCout) const override;

public:
  ///   get / set the pointer to a L1GtCondition
  inline const L1GtCastorTemplate *gtCastorTemplate() const { return m_gtCastorTemplate; }

  void setGtCastorTemplate(const L1GtCastorTemplate *);

  ///   get / set the result
  inline const bool conditionResult() const { return m_conditionResult; }

  inline void setConditionResult(const bool result) { m_conditionResult = result; }

private:
  /// copy function for copy constructor and operator=
  void copy(const L1GtCastorCondition &cp);

private:
  /// pointer to a L1GtCastorTemplate
  const L1GtCastorTemplate *m_gtCastorTemplate;

  /// condition result
  bool m_conditionResult;
};

#endif
