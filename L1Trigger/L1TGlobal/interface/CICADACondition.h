#ifndef L1Trigger_L1TGlobal_CICADACondition_h
#define L1Trigger_L1TGlobal_CICADACondition_h

#include <iosfwd>
#include <string>

#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

class GlobalCondition;
class CICADATemplate;

namespace l1t {
  class GlobalBoard;

  class CICADACondition : public ConditionEvaluation {
  public:
    CICADACondition();
    CICADACondition(const GlobalCondition*, const GlobalBoard*);
    CICADACondition(const CICADACondition&);
    ~CICADACondition() override = default;

    CICADACondition& operator=(const CICADACondition&);

    const bool evaluateCondition(const int bxEval) const override;

    void print(std::ostream& myCout) const override;

    inline const CICADATemplate* gtCICADATemplate() const { return m_gtCICADATemplate; }

    void setGtCICADATemplate(const CICADATemplate* cicadaTemplate) { m_gtCICADATemplate = cicadaTemplate; }

    inline const GlobalBoard* getuGtB() const { return m_uGtB; }

    void setuGtB(const GlobalBoard* ptrGTB) { m_uGtB = ptrGTB; }

  private:
    void copy(const CICADACondition& cp);

    const CICADATemplate* m_gtCICADATemplate;

    const GlobalBoard* m_uGtB;
  };
}  // namespace l1t
#endif
