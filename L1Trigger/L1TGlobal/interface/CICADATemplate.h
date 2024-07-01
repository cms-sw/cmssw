#ifndef L1Trigger_L1TGlobal_CICADATemplate_h
#define L1Trigger_L1TGlobal_CICADATemplate_h

#include <string>
#include <iosfwd>

#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

class CICADATemplate : public GlobalCondition {
public:
  CICADATemplate();
  CICADATemplate(const std::string&);
  CICADATemplate(const std::string&, const l1t::GtConditionType&);
  CICADATemplate(const CICADATemplate&);
  ~CICADATemplate() = default;

  CICADATemplate& operator=(const CICADATemplate&);

  struct ObjectParameter {
    float minCICADAThreshold;
    float maxCICADAThreshold;
  };
  inline const std::vector<ObjectParameter>* objectParameter() const { return &m_objectParameter; }

  void setConditionParameter(const std::vector<ObjectParameter>& objParameter) { m_objectParameter = objParameter; }

  void print(std::ostream& myCout) const override;

  friend std::ostream& operator<<(std::ostream&, const CICADATemplate&);

private:
  void copy(const CICADATemplate& cp);
  std::vector<ObjectParameter> m_objectParameter;
};

#endif
