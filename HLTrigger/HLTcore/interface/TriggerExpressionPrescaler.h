#ifndef HLTrigger_HLTcore_TriggerExpressionPrescaler_h
#define HLTrigger_HLTcore_TriggerExpressionPrescaler_h

#include <vector>

#include "HLTrigger/HLTcore/interface/TriggerExpressionOperators.h"

namespace triggerExpression {

  class Prescaler : public UnaryOperator {
  public:
    Prescaler(Evaluator* arg, unsigned int prescale);
    ~Prescaler() override;

    bool operator()(const Data& data) const override;

    void init(const Data& data) override;

    void dump(std::ostream& out, bool const ignoreMasks = false) const override {
      out << '(';
      m_arg->dump(out, ignoreMasks);
      out << " / " << m_prescale << ')';
    }

  private:
    // return the counter of the stream of the current event, initialised to the event number on first use
    unsigned int& counter(const Data& data) const;

    unsigned int m_prescale;

    // one event counter per stream if the expression is shared by several streams, otherwise a single counter;
    // the vector is sized by init(), each element is then used only by its own stream; Counter is defined in
    // TriggerExpressionPrescaler.cc, because its alignment depends on the compiler flags
    struct Counter;
    mutable std::vector<Counter> m_counters;
  };

}  // namespace triggerExpression

#endif  // HLTrigger_HLTcore_TriggerExpressionPrescaler_h
