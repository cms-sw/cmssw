#ifndef L1Trigger_L1TGlobal_AXOL1TLTemplate_h
#define L1Trigger_L1TGlobal_AXOL1TLTemplate_h

/**
 * \class AXOL1TLTemplate
 *
 *
 * Description: L1 Global Trigger AXOL1TL template.
 *
 * \author: Melissa Quinnan (UC San Diego)
 *
 */

// system include files
#include <string>
#include <iosfwd>

// user include files

//   base class
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

// forward declarations

// class declaration
class AXOL1TLTemplate : public GlobalCondition {
public:
  // constructor
  AXOL1TLTemplate();

  // constructor
  AXOL1TLTemplate(const std::string&);

  // constructor
  AXOL1TLTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  AXOL1TLTemplate(const AXOL1TLTemplate&);

  // destructor
  ~AXOL1TLTemplate() override;

  // assign operator
  AXOL1TLTemplate& operator=(const AXOL1TLTemplate&);

  // typedef for a single object template
  struct ObjectParameter {
    int minAXOL1TLThreshold;
    int maxAXOL1TLThreshold;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>& objParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const AXOL1TLTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const AXOL1TLTemplate& cp);

  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
};

#endif
