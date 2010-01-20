#ifndef HLTrigger_HLTfilters_ExpressionHLTReader_h
#define HLTrigger_HLTfilters_ExpressionHLTReader_h

#include <vector>
#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

namespace edm {
  class TriggerNames ;
}

namespace hlt {

class TriggerExpressionHLTReader : public TriggerExpressionEvaluator {
public:
  TriggerExpressionHLTReader(const std::string & pattern) :
    m_pattern(pattern),
    m_triggers(),
    m_throw(false)
  { }

  void init(const edm::TriggerNames & triggerNames);

  bool operator()(const TriggerExpressionData & data);
  
  void dump(std::ostream & out) const;

private:
  std::string               m_pattern;
  std::vector<unsigned int> m_triggers;
  bool                      m_throw;
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionHLTReader_h
