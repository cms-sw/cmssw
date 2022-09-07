#ifndef HLTrigger_HLTcore_TriggerExpressionPrescaler_h
#define HLTrigger_HLTcore_TriggerExpressionPrescaler_h

#include "HLTrigger/HLTcore/interface/TriggerExpressionOperators.h"

namespace triggerExpression {

  class Prescaler : public UnaryOperator {
  public:
    Prescaler(Evaluator* arg, unsigned int prescale) : UnaryOperator(arg), m_prescale(prescale), m_counter() {}

    bool operator()(const Data& data) const override;

    void init(const Data& data) override;

    void dump(std::ostream& out, bool const ignoreMasks = false) const override {
      out << '(';
      m_arg->dump(out, ignoreMasks);
      out << " / " << m_prescale << ')';
    }

  private:
    unsigned int m_prescale;
    mutable unsigned int m_counter;
  };

}  // namespace triggerExpression

#endif  // HLTrigger_HLTcore_TriggerExpressionPrescaler_h
