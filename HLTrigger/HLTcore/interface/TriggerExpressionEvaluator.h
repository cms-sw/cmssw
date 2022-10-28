#ifndef HLTrigger_HLTcore_TriggerExpressionEvaluator_h
#define HLTrigger_HLTcore_TriggerExpressionEvaluator_h

#include <iostream>
#include <string>
#include <vector>
#include <utility>

namespace triggerExpression {

  class Data;

  class Evaluator {
  public:
    Evaluator() = default;

    // virtual destructor
    virtual ~Evaluator() = default;

    // check if the data satisfies the logical expression
    virtual bool operator()(const Data& data) const = 0;

    // (re)initialise the logical expression
    virtual void init(const Data& data) {}

    // list CMSSW path patterns associated to the logical expression
    virtual std::vector<std::string> patterns() const { return {}; }

    // list of triggers associated to the Evaluator (filled only for certain derived classes)
    virtual std::vector<std::pair<std::string, unsigned int>> triggers() const { return {}; }

    // dump the logical expression to the output stream
    virtual void dump(std::ostream& out, bool const ignoreMasks = false) const = 0;

    // apply masks based on another Evaluator
    virtual void mask(Evaluator const&) {}

    // methods to control m_masksEnabled boolean
    virtual bool masksEnabled() const { return m_masksEnabled; }
    virtual void enableMasks() { m_masksEnabled = true; }
    virtual void disableMasks() { m_masksEnabled = false; }

  private:
    bool m_masksEnabled = false;
  };

  inline std::ostream& operator<<(std::ostream& out, const Evaluator& eval) {
    eval.dump(out);
    return out;
  }

}  // namespace triggerExpression

#endif  // HLTrigger_HLTcore_TriggerExpressionEvaluator_h
