#ifndef HLTrigger_HLTfilters_TriggerExpressionPrescaler_h
#define HLTrigger_HLTfilters_TriggerExpressionPrescaler_h

#include "HLTrigger/HLTcore/interface/TriggerExpressionOperators.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

namespace triggerExpression {

  class Prescaler : public UnaryOperator {
  public:
    Prescaler(Evaluator* arg, unsigned int prescale) : UnaryOperator(arg), m_prescale(prescale), m_counter() {}

    bool operator()(const Data& data) const override;

    void init(const Data& data) override;

    void dump(std::ostream& out) const override { out << "(" << (*m_arg) << " / " << m_prescale << ")"; }

  private:
    unsigned int m_prescale;
    mutable unsigned int m_counter;
  };

}  // namespace triggerExpression

#endif  // HLTrigger_HLTfilters_TriggerExpressionPrescaler_h
