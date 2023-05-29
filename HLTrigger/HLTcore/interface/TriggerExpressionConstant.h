#ifndef HLTrigger_HLTcore_TriggerExpressionConstant_h
#define HLTrigger_HLTcore_TriggerExpressionConstant_h

#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

  class Data;

  class Constant : public Evaluator {
  public:
    Constant(bool value) : m_value(value) {}

    bool operator()(const Data& data) const override { return m_value; }

    void dump(std::ostream& out, bool const ignoreMasks = false) const override { out << (m_value ? "TRUE" : "FALSE"); }

  private:
    bool m_value;
  };

}  // namespace triggerExpression

#endif  // HLTrigger_HLTcore_TriggerExpressionConstant_h
