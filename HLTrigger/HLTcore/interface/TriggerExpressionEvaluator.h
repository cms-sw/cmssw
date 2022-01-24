#ifndef HLTrigger_HLTfilters_TriggerExpressionEvaluator_h
#define HLTrigger_HLTfilters_TriggerExpressionEvaluator_h

#include <iostream>
#include <string>
#include <vector>

namespace triggerExpression {

  class Data;

  class Evaluator {
  public:
    Evaluator() = default;

    // check if the data satisfies the logical expression
    virtual bool operator()(const Data& data) const = 0;

    // (re)initialise the logical expression
    virtual void init(const Data& data) {}

    // list CMSSW path patterns associated to the logical expression
    virtual std::vector<std::string> patterns() const { return {}; }

    // dump the logical expression to the output stream
    virtual void dump(std::ostream& out) const = 0;

    // virtual destructor
    virtual ~Evaluator() = default;
  };

  inline std::ostream& operator<<(std::ostream& out, const Evaluator& eval) {
    eval.dump(out);
    return out;
  }

}  // namespace triggerExpression

#endif  // HLTrigger_HLTfilters_TriggerExpressionEvaluator_h
