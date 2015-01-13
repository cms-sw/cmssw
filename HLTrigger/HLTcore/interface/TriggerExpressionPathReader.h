#ifndef HLTrigger_HLTfilters_TriggerExpressionPathReader_h
#define HLTrigger_HLTfilters_TriggerExpressionPathReader_h

#include <vector>
#include <string>

#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

class PathReader : public Evaluator {
public:
  PathReader(const std::string & pattern) :
    m_pattern(pattern),
    m_triggers()
  { }

  bool operator()(const Data & data) const;

  void init(const Data & data);

  void dump(std::ostream & out) const;

private:
  std::string m_pattern;
  std::vector<std::pair<std::string, unsigned int> > m_triggers;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionPathReader_h
