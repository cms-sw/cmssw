#ifndef HLTrigger_HLTfilters_ExpressionPrescaler_h
#define HLTrigger_HLTfilters_ExpressionPrescaler_h

#include "HLTrigger/HLTfilters/interface/TriggerExpressionOperators.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionCache.h"

namespace hlt {

class Prescaler : public UnaryOperator {
public:
  Prescaler(TriggerExpressionEvaluator * arg, unsigned int prescale) :
    UnaryOperator(arg),
    m_prescale(prescale),
    m_counter()
  { }

  bool operator()(const TriggerExpressionCache & data) {
    // initialize the counter to the first event number seen, 
    // in order to avoid all prescalers on different FUs to be syncronous
    if (m_counter == 0)
      m_counter = data.eventNumber();

    ++m_counter;
    if (m_prescale == 0)
      return false;

    if (m_prescale == 1)
      return true;

    return ((m_counter % m_prescale) == 0);
  }

private:
  unsigned int m_prescale;
  unsigned int m_counter;
};

} // namespace hlt

#endif // HLTrigger_HLTfilters_ExpressionPrescaler_h
