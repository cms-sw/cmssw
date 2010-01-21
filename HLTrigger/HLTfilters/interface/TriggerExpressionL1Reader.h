#ifndef HLTrigger_HLTfilters_TriggerExpressionL1Reader_h
#define HLTrigger_HLTfilters_TriggerExpressionL1Reader_h

#include <vector>
#include <string>

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"

class L1GtTriggerMenu;
class L1GtTriggerMask;

namespace triggerExpression {

class L1Reader : public Evaluator {
public:
  L1Reader(const std::string & pattern) :
    m_ignoreMask(false),
    m_daqPartitions(0x00),
    m_throw(false),
    m_pattern(pattern),
    m_triggers()
  { }

  void init(const L1GtTriggerMenu & menu, const L1GtTriggerMask & mask);

  bool operator()(const Data & data);
  
  void dump(std::ostream & out) const;

private:
  bool         m_ignoreMask;
  unsigned int m_daqPartitions;
  bool         m_throw;
  std::string  m_pattern;
  std::vector<std::pair<std::string, unsigned int> > m_triggers;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionL1Reader_h
