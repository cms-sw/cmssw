#ifndef L1Trigger_L1TGlobal_EnergySumZdcTemplate_h
#define L1Trigger_L1TGlobal_EnergySumZdcTemplate_h

/**
 * \class EnergySumZdcTemplate
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
class EnergySumZdcTemplate : public GlobalCondition {
public:
  // constructor
  EnergySumZdcTemplate();

  // constructor
  EnergySumZdcTemplate(const std::string&);

  // constructor
  EnergySumZdcTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  EnergySumZdcTemplate(const EnergySumZdcTemplate&);

  // destructor
  ~EnergySumZdcTemplate() override;

  // assign operator
  EnergySumZdcTemplate& operator=(const EnergySumZdcTemplate&);

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
  friend std::ostream& operator<<(std::ostream&, const EnergySumZdcTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const EnergySumZdcTemplate& cp);

private:
  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
};

#endif
