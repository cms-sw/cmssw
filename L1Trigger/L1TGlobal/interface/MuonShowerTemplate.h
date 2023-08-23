#ifndef L1Trigger_L1TGlobal_MuonShowerTemplate_h
#define L1Trigger_L1TGlobal_MuonShowerTemplate_h

/**
 * \class MuonShowerTemplate
 *
 *
 * Description: L1 Global Trigger muon shower template.
 *
 * \author: Sven Dildick (Rice University)
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
class MuonShowerTemplate : public GlobalCondition {
public:
  // constructor
  MuonShowerTemplate();

  // constructor
  MuonShowerTemplate(const std::string&);

  // constructor
  MuonShowerTemplate(const std::string&, const l1t::GtConditionType&);

  // copy constructor
  MuonShowerTemplate(const MuonShowerTemplate&);

  // destructor
  ~MuonShowerTemplate() override;

  // assign operator
  MuonShowerTemplate& operator=(const MuonShowerTemplate&);

  // typedef for a single object template
  struct ObjectParameter {
    bool MuonShower0;
    bool MuonShower1;
    bool MuonShower2;
    bool MuonShowerOutOfTime0;
    bool MuonShowerOutOfTime1;
  };

public:
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  /// set functions
  void setConditionParameter(const std::vector<ObjectParameter>& objParameter);

  /// print the condition
  void print(std::ostream& myCout) const override;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const MuonShowerTemplate&);

private:
  /// copy function for copy constructor and operator=
  void copy(const MuonShowerTemplate& cp);

  /// variables containing the parameters
  std::vector<ObjectParameter> m_objectParameter;
};

#endif
