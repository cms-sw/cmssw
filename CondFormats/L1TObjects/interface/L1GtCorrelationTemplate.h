#ifndef CondFormats_L1TObjects_L1GtCorrelationTemplate_h
#define CondFormats_L1TObjects_L1GtCorrelationTemplate_h

/**
 * \class L1GtCorrelationTemplate
 *
 *
 * Description: L1 Global Trigger correlation template.
 * Includes spatial correlation for two objects of different type.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <iosfwd>

// user include files

//   base class
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// class declaration
class L1GtCorrelationTemplate : public L1GtCondition {
public:
  /// constructor(s)
  ///   default
  L1GtCorrelationTemplate();

  ///   from condition name
  L1GtCorrelationTemplate(const std::string&);

  ///   from condition name, the category of first sub-condition, the category of the
  ///   second sub-condition, the index of first sub-condition in the cor* vector,
  ///   the index of second sub-condition in the cor* vector
  L1GtCorrelationTemplate(
      const std::string&, const L1GtConditionCategory&, const L1GtConditionCategory&, const int, const int);

  /// copy constructor
  L1GtCorrelationTemplate(const L1GtCorrelationTemplate&);

  /// destructor
  ~L1GtCorrelationTemplate() override;

  /// assign operator
  L1GtCorrelationTemplate& operator=(const L1GtCorrelationTemplate&);

public:
  /// typedef for correlation parameters
  struct CorrelationParameter {
    std::string deltaEtaRange;

    std::string deltaPhiRange;
    unsigned int deltaPhiMaxbits;

    COND_SERIALIZABLE;
  };

public:
  /// get / set the category of the two sub-conditions
  inline const L1GtConditionCategory cond0Category() const { return m_cond0Category; }

  inline const L1GtConditionCategory cond1Category() const { return m_cond1Category; }

  void setCond0Category(const L1GtConditionCategory&);
  void setCond1Category(const L1GtConditionCategory&);

  /// get / set the index of the two sub-conditions in the cor* vector from menu
  inline const int cond0Index() const { return m_cond0Index; }

  inline const int cond1Index() const { return m_cond1Index; }

  void setCond0Index(const int&);
  void setCond1Index(const int&);

  /// get / set correlation parameters

  inline const CorrelationParameter* correlationParameter() const { return &m_correlationParameter; }

  void setCorrelationParameter(const CorrelationParameter& corrParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtCorrelationTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const L1GtCorrelationTemplate& cp);

private:
  L1GtConditionCategory m_cond0Category;
  L1GtConditionCategory m_cond1Category;
  int m_cond0Index;
  int m_cond1Index;
  CorrelationParameter m_correlationParameter;

  COND_SERIALIZABLE;
};

#endif
