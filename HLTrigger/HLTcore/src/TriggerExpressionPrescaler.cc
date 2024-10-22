#include "HLTrigger/HLTcore/interface/TriggerExpressionPrescaler.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

namespace triggerExpression {

  bool Prescaler::operator()(const Data& data) const {
    // if the prescale factor is 0, we never need to run any dependent module,
    // so we can safely skip the rest of the processing
    if (m_prescale == 0)
      return false;

    bool result = ((*m_arg)(data));
    if (not result)
      return false;

    // if the prescale factor is 1, we do not need to keep track of the event counter
    if (m_prescale == 1)
      return true;

    return (++m_counter % m_prescale) == 0;
  }

  void Prescaler::init(const Data& data) {
    // initialize the depending modules
    UnaryOperator::init(data);

    // initialize the counter to the first event number seen,
    // in order to avoid all prescalers on different FUs to be synchronous
    m_counter = data.eventNumber();
  }

}  // namespace triggerExpression
