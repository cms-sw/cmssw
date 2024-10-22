#ifndef HLTrigger_HLTcore_TriggerExpressionPathReader_h
#define HLTrigger_HLTcore_TriggerExpressionPathReader_h

#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

  class PathReader : public Evaluator {
  public:
    PathReader(const std::string& pattern)
        : m_pattern{pattern}, m_triggers{}, m_triggersAfterMasking{}, m_initialised{false} {}

    bool operator()(const Data& data) const override;

    void init(const Data& data) override;

    std::vector<std::string> patterns() const override { return {m_pattern}; }

    void dump(std::ostream& out, bool const ignoreMasks = false) const override;

    void mask(Evaluator const& eval) override;

    std::vector<std::pair<std::string, unsigned int>> triggers() const override { return m_triggers; }
    std::vector<std::pair<std::string, unsigned int>> triggersAfterMasking() const { return m_triggersAfterMasking; }

  private:
    std::string m_pattern;
    std::vector<std::pair<std::string, unsigned int>> m_triggers;
    std::vector<std::pair<std::string, unsigned int>> m_triggersAfterMasking;
    bool m_initialised;
  };

}  // namespace triggerExpression

#endif  // HLTrigger_HLTcore_TriggerExpressionPathReader_h
