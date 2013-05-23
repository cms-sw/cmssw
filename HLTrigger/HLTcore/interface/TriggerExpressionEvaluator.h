#ifndef HLTrigger_HLTfilters_TriggerExpressionEvaluator_h
#define HLTrigger_HLTfilters_TriggerExpressionEvaluator_h

#include <iostream>

namespace triggerExpression {

class Data;

class Evaluator {
public:
  Evaluator() { }

  // pure virtual, need a concrete implementation
  virtual bool operator()(const Data & data) const = 0;

  // virtual function, do nothing unless overridden
  virtual void init(const Data & data) { }

  // pure virtual, need a concrete implementation
  virtual void dump(std::ostream & out) const = 0;

  // virtual destructor
  virtual ~Evaluator() { }
};

inline 
std::ostream & operator<<(std::ostream & out, const Evaluator & eval) {
  eval.dump(out);
  return out;
}

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionEvaluator_h
