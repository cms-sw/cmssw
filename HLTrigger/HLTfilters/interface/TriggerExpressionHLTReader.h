#ifndef HLTrigger_HLTfilters_TriggerExpressionHLTReader_h
#define HLTrigger_HLTfilters_TriggerExpressionHLTReader_h

#include <vector>
#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace edm {
  class TriggerNames ;
}

namespace triggerExpression {

class HLTReader : public Evaluator {
public:
  HLTReader(const std::string & pattern) :
    m_pattern(pattern),
    m_triggers(),
    m_throw(false)
  { }

  void init(const edm::TriggerNames & triggerNames);

  bool operator()(const Data & data);
  
  void dump(std::ostream & out) const;

private:
  std::string               m_pattern;
  std::vector<unsigned int> m_triggers;
  bool                      m_throw;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionHLTReader_h
