#ifndef L1Trigger_L1TGlobal_CorrelationWithOverlapRemovalTemplate_h
#define L1Trigger_L1TGlobal_CorrelationWithOverlapRemovalTemplate_h

/**
 * \class CorrelationWithOverlapRemovalTemplate,  motivated by class CorrealtionTemplate
 *
 *
 * Description: L1 Global Trigger correlation with overlap removal template.
 * Includes spatial correlation for two objects and removal of overlap with third object
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vladimir Rekovic
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
class CorrelationWithOverlapRemovalTemplate : public GlobalCondition {
public:
  /// constructor(s)
  ///   default
  CorrelationWithOverlapRemovalTemplate();

  ///   from condition name
  CorrelationWithOverlapRemovalTemplate(const std::string&);

  ///   from condition name, the category of first sub-condition, the category of the
  ///   second sub-condition, the category of third sub-condition, the index of first sub-condition in the cor* vector,
  ///   the index of second sub-condition in the cor* vector, the index of second sub-condition in the cor* vector
  CorrelationWithOverlapRemovalTemplate(const std::string&,
                                        const l1t::GtConditionCategory&,
                                        const l1t::GtConditionCategory&,
                                        const l1t::GtConditionCategory&,
                                        const int,
                                        const int,
                                        const int);

  /// copy constructor
  CorrelationWithOverlapRemovalTemplate(const CorrelationWithOverlapRemovalTemplate&);

  /// destructor
  ~CorrelationWithOverlapRemovalTemplate() override;

  /// assign operator
  CorrelationWithOverlapRemovalTemplate& operator=(const CorrelationWithOverlapRemovalTemplate&);

public:
  /// typedef for correlation parameters
  struct CorrelationWithOverlapRemovalParameter {
    // Cut values in hardware
    long long minEtaCutValue;
    long long maxEtaCutValue;
    unsigned int precEtaCut;

    long long minPhiCutValue;
    long long maxPhiCutValue;
    unsigned int precPhiCut;

    long long minDRCutValue;
    long long maxDRCutValue;

    unsigned int precDRCut;

    long long minMassCutValue;
    long long maxMassCutValue;
    unsigned int precMassCut;

    long long minTBPTCutValue;
    long long maxTBPTCutValue;
    unsigned int precTBPTCut;

    long long minOverlapRemovalEtaCutValue;
    long long maxOverlapRemovalEtaCutValue;
    unsigned int precOverlapRemovalEtaCut;

    long long minOverlapRemovalPhiCutValue;
    long long maxOverlapRemovalPhiCutValue;
    unsigned int precOverlapRemovalPhiCut;

    long long minOverlapRemovalDRCutValue;
    long long maxOverlapRemovalDRCutValue;
    unsigned int precOverlapRemovalDRCut;

    //Requirement on charge of legs (currently only Mu-Mu).
    unsigned int chargeCorrelation;

    int corrCutType;
  };

public:
  /// get / set the category of the thre sub-conditions
  inline const l1t::GtConditionCategory cond0Category() const { return m_cond0Category; }

  inline const l1t::GtConditionCategory cond1Category() const { return m_cond1Category; }

  inline const l1t::GtConditionCategory cond2Category() const { return m_cond2Category; }

  void setCond0Category(const l1t::GtConditionCategory&);
  void setCond1Category(const l1t::GtConditionCategory&);
  void setCond2Category(const l1t::GtConditionCategory&);

  /// get / set the index of the two sub-conditions in the cor* vector from menu
  inline const int cond0Index() const { return m_cond0Index; }

  inline const int cond1Index() const { return m_cond1Index; }

  inline const int cond2Index() const { return m_cond2Index; }

  void setCond0Index(const int&);
  void setCond1Index(const int&);
  void setCond2Index(const int&);

  /// get / set correlation parameters

  inline const CorrelationWithOverlapRemovalParameter* correlationParameter() const { return &m_correlationParameter; }

  void setCorrelationWithOverlapRemovalParameter(const CorrelationWithOverlapRemovalParameter& corrParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const CorrelationWithOverlapRemovalTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const CorrelationWithOverlapRemovalTemplate& cp);

private:
  l1t::GtConditionCategory m_cond0Category;
  l1t::GtConditionCategory m_cond1Category;
  l1t::GtConditionCategory m_cond2Category;
  int m_cond0Index;
  int m_cond1Index;
  int m_cond2Index;
  CorrelationWithOverlapRemovalParameter m_correlationParameter;
};

#endif
