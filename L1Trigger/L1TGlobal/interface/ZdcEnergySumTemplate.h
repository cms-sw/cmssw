#ifndef L1Trigger_L1TGlobal_ZdcEnergySumTemplate_h
#define L1Trigger_L1TGlobal_ZdcEnergySumTemplate_h

/**
 * \class ZdcEnergySumTemplate
 *
 *
 * Description: L1 Global Trigger energy-sum template.
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
#include <string>
#include <iosfwd>

// user include files

//   base class
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

// forward declarations

// class declaration
class ZdcEnergySumTemplate : public GlobalCondition {
public:
  // constructor
  ZdcEnergySumTemplate();

  // constructor
  ZdcEnergySumTemplate(const std::string&);

  // constructor
  ZdcEnergySumTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  ZdcEnergySumTemplate(const ZdcEnergySumTemplate&);

  // destructor
  ~ZdcEnergySumTemplate() override;

  // assign operator
  ZdcEnergySumTemplate& operator=(const ZdcEnergySumTemplate&);

public:
  struct ObjectParameter {
    unsigned int etLowThreshold;
    unsigned int etHighThreshold;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>&);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const ZdcEnergySumTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const ZdcEnergySumTemplate& cp);

private:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
};

#endif
