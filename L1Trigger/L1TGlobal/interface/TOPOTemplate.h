#ifndef L1Trigger_L1TGlobal_TOPOTemplate_h
#define L1Trigger_L1TGlobal_TOPOTemplate_h

/**
 * \class TOPOTemplate
 *
 *
 * Description: L1 Global Trigger TOPO template.
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
class TOPOTemplate : public GlobalCondition {
public:
  // constructor
  TOPOTemplate();

  // constructor
  TOPOTemplate(const std::string&);

  // constructor
  TOPOTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  TOPOTemplate(const TOPOTemplate&);

  // destructor
  ~TOPOTemplate() override;

  // assign operator
  TOPOTemplate& operator=(const TOPOTemplate&);

  // typedef for a single object template
  struct ObjectParameter {
    int minTOPOThreshold;
    int maxTOPOThreshold;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  inline const std::string& modelVersion() const { return m_modelVersion; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>& objParameter);

  void setModelVersion(const std::string& modelversion);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const TOPOTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const TOPOTemplate& cp);

  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;

  /// model version
  std::string m_modelVersion;
};

#endif
