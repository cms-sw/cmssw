#ifndef HLTrigger_HLTfilters_TriggerExpressionL1TechReader_h
#define HLTrigger_HLTfilters_TriggerExpressionL1TechReader_h

#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace triggerExpression {

class L1TechReader : public Evaluator {
public:
  L1TechReader(const std::string & pattern) :
    m_pattern(pattern)
  { }

  bool operator()(const Data & data);
  
  void dump(std::ostream & out) const;

private:
  std::string m_pattern;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionL1TechReader_h
