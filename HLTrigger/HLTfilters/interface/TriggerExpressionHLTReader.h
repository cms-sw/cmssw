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
    m_throw(false),
    m_pattern(pattern),
    m_triggers()
  { }

  void init(const edm::TriggerNames & triggerNames);

  bool operator()(const Data & data);
  
  void dump(std::ostream & out) const;

private:
  bool        m_throw;
  std::string m_pattern;
  std::vector<std::pair<std::string, unsigned int> > m_triggers;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionHLTReader_h
