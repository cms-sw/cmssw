#ifndef HLTrigger_HLTfilters_TriggerExpressionEvaluator_h
#define HLTrigger_HLTfilters_TriggerExpressionEvaluator_h

#include <iostream>

namespace edm {
  class Event;
  class EventSetup;
} // namespace edm

namespace triggerExpression {

class Data;

class Evaluator {
public:
  Evaluator() { }

  // the default implementation does nothing
  virtual void configure(const edm::EventSetup &) { }

  // pure virtual, need a concrete implementation
  virtual bool operator()(const Data &) = 0;

  // pure virtual, need a concrete implementation
  virtual void dump(std::ostream &) const = 0;
};

inline 
std::ostream & operator<<(std::ostream & out, const Evaluator & eval) {
  eval.dump(out);
  return out;
}

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionEvaluator_h
