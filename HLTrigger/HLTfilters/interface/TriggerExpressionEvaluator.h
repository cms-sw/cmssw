#ifndef HLTrigger_HLTfilters_ExpressionEvaluator_h
#define HLTrigger_HLTfilters_ExpressionEvaluator_h

#include <iostream>

namespace edm {
  class Event;
  class EventSetup;
} // namespace edm

namespace hlt {

class TriggerExpressionCache;

class TriggerExpressionEvaluator {
public:
  TriggerExpressionEvaluator() { }

  // the default implementation does nothing
  virtual void configure(const edm::EventSetup &) { }

  // pure virtual, need a concrete implementation
  virtual bool operator()(const TriggerExpressionCache &) = 0;

  // pure virtual, need a concrete implementation
  virtual void dump(std::ostream &) = 0;
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionEvaluator_h
