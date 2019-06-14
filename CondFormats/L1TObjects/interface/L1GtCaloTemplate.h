#ifndef CondFormats_L1TObjects_L1GtCaloTemplate_h
#define CondFormats_L1TObjects_L1GtCaloTemplate_h

/**
 * \class L1GtCaloTemplate
 *
 *
 * Description: L1 Global Trigger calo template.
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

// forward declarations

// class declaration
class L1GtCaloTemplate : public L1GtCondition {
public:
  // constructor
  L1GtCaloTemplate();

  // constructor
  L1GtCaloTemplate(const std::string&);

  // constructor
  L1GtCaloTemplate(const std::string&, const L1GtConditionType&);

  // copy constructor
  L1GtCaloTemplate(const L1GtCaloTemplate&);

  // destructor
  ~L1GtCaloTemplate() override;

  // assign operator
  L1GtCaloTemplate& operator=(const L1GtCaloTemplate&);

public:
  /// typedef for a single object template
  struct ObjectParameter {
    unsigned int etThreshold;
    unsigned int etaRange;
    unsigned int phiRange;

    COND_SERIALIZABLE;
  };

  /// typedef for correlation parameters
  struct CorrelationParameter {
    unsigned long long deltaEtaRange;

    unsigned long long deltaPhiRange;
    unsigned int deltaPhiMaxbits;

    COND_SERIALIZABLE;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  inline const CorrelationParameter* correlationParameter() const { return &m_correlationParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>& objParameter,
                             const CorrelationParameter& corrParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtCaloTemplate&);

protected:
  /// copy function for copy constructor and operator=
  void copy(const L1GtCaloTemplate& cp);

protected:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
  CorrelationParameter m_correlationParameter;

  COND_SERIALIZABLE;
};

#endif
