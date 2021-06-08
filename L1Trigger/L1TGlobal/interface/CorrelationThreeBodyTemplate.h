#ifndef L1Trigger_L1TGlobal_CorrelationThreeBodyTemplate_h
#define L1Trigger_L1TGlobal_CorrelationThreeBodyTemplate_h

/**
 * \class CorrelationThreeBodyTemplate
 *
 *
 * Description: L1 Global Trigger three-body correlation template:
 *              include invariant mass calculation for three-muon events
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Elisa Fontanesi - Boston University 
 *          Starting from CorrelationTemplate.h written by Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <iosfwd>

// user include files

//   base class
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"
#include "L1Trigger/L1TGlobal/interface/GlobalDefinitions.h"

// forward declarations

// class declaration
class CorrelationThreeBodyTemplate : public GlobalCondition {
public:
  /// constructor(s)
  ///   default
  CorrelationThreeBodyTemplate();

  ///   from condition name
  CorrelationThreeBodyTemplate(const std::string&);

  ///   from condition name, the category of first, second, and third subcondition,
  ///   the index of first, second, third subcondition in the cor* vector
  CorrelationThreeBodyTemplate(const std::string&,
                               const l1t::GtConditionCategory&,
                               const l1t::GtConditionCategory&,
                               const l1t::GtConditionCategory&,
                               const int,
                               const int,
                               const int);

  /// copy constructor
  CorrelationThreeBodyTemplate(const CorrelationThreeBodyTemplate&);

  /// destructor
  ~CorrelationThreeBodyTemplate() override;

  /// assign operator
  CorrelationThreeBodyTemplate& operator=(const CorrelationThreeBodyTemplate&);

public:
  /// typedef for correlation three-body parameters
  struct CorrelationThreeBodyParameter {
    //Cut values in hardware
    long long minEtaCutValue;
    long long maxEtaCutValue;
    unsigned int precEtaCut;

    long long minPhiCutValue;
    long long maxPhiCutValue;
    unsigned int precPhiCut;

    long long minMassCutValue;
    long long maxMassCutValue;
    unsigned int precMassCut;

    int corrCutType;
  };

public:
  /// get / set the category of the three subconditions
  inline const l1t::GtConditionCategory cond0Category() const { return m_cond0Category; }
  inline const l1t::GtConditionCategory cond1Category() const { return m_cond1Category; }
  inline const l1t::GtConditionCategory cond2Category() const { return m_cond2Category; }

  void setCond0Category(const l1t::GtConditionCategory&);
  void setCond1Category(const l1t::GtConditionCategory&);
  void setCond2Category(const l1t::GtConditionCategory&);

  /// get / set the index of the three subconditions in the cor* vector from menu
  inline const int cond0Index() const { return m_cond0Index; }
  inline const int cond1Index() const { return m_cond1Index; }
  inline const int cond2Index() const { return m_cond2Index; }

  void setCond0Index(const int&);
  void setCond1Index(const int&);
  void setCond2Index(const int&);

  /// get / set correlation parameters
  inline const CorrelationThreeBodyParameter* correlationThreeBodyParameter() const {
    return &m_correlationThreeBodyParameter;
  }
  void setCorrelationThreeBodyParameter(const CorrelationThreeBodyParameter& corrThreeBodyParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const CorrelationThreeBodyTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const CorrelationThreeBodyTemplate& cp);

private:
  l1t::GtConditionCategory m_cond0Category;
  l1t::GtConditionCategory m_cond1Category;
  l1t::GtConditionCategory m_cond2Category;
  int m_cond0Index;
  int m_cond1Index;
  int m_cond2Index;
  CorrelationThreeBodyParameter m_correlationThreeBodyParameter;
};

#endif
