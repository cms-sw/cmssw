#ifndef HLTrigger_HLTfilters_ExpressionL1TechReader_h
#define HLTrigger_HLTfilters_ExpressionL1TechReader_h

#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace hlt {

class TriggerExpressionL1TechReader : public TriggerExpressionEvaluator {
public:
  TriggerExpressionL1TechReader(const std::string & pattern) :
    m_pattern(pattern)
  { }

  bool operator()(const TriggerExpressionCache & data);

private:
  std::string m_pattern;
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionL1TechReader_h
