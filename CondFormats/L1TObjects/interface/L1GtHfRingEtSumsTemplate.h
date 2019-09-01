#ifndef CondFormats_L1TObjects_L1GtHfRingEtSumsTemplate_h
#define CondFormats_L1TObjects_L1GtHfRingEtSumsTemplate_h

/**
 * \class L1GtHfRingEtSumsTemplate
 *
 *
 * Description: L1 Global Trigger "HF Ring ET sums" template.
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
class L1GtHfRingEtSumsTemplate : public L1GtCondition {
public:
  // constructor
  L1GtHfRingEtSumsTemplate();

  // constructor
  L1GtHfRingEtSumsTemplate(const std::string&);

  // constructor
  L1GtHfRingEtSumsTemplate(const std::string&, const L1GtConditionType&);

  // copy constructor
  L1GtHfRingEtSumsTemplate(const L1GtHfRingEtSumsTemplate&);

  // destructor
  ~L1GtHfRingEtSumsTemplate() override;

  // assign operator
  L1GtHfRingEtSumsTemplate& operator=(const L1GtHfRingEtSumsTemplate&);

public:
  /// typedef for a single object template
  struct ObjectParameter {
    unsigned int etSumIndex;
    unsigned int etSumThreshold;

    COND_SERIALIZABLE;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>&);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtHfRingEtSumsTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const L1GtHfRingEtSumsTemplate& cp);

private:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;

  COND_SERIALIZABLE;
};

#endif
