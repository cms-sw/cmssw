#ifndef HLTrigger_HLTfilters_ExpressionL1Reader_h
#define HLTrigger_HLTfilters_ExpressionL1Reader_h

#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace hlt {

class TriggerExpressionL1Reader : public TriggerExpressionEvaluator {
public:
  TriggerExpressionL1Reader(const std::string & pattern) :
    m_pattern(pattern)
  { }

  bool operator()(const TriggerExpressionData & data);
  
  void dump(std::ostream & out) const;

private:
  std::string m_pattern;
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionL1Reader_h
