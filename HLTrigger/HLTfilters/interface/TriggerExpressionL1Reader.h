#ifndef HLTrigger_HLTfilters_TriggerExpressionL1Reader_h
#define HLTrigger_HLTfilters_TriggerExpressionL1Reader_h

#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

class L1Reader : public Evaluator {
public:
  L1Reader(const std::string & pattern) :
    m_pattern(pattern)
  { }

  bool operator()(const Data & data);
  
  void dump(std::ostream & out) const;

private:
  std::string m_pattern;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionL1Reader_h
