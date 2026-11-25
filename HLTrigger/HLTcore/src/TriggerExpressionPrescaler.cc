#include <cassert>
#include <new>

#include "HLTrigger/HLTcore/interface/TriggerExpressionPrescaler.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

namespace triggerExpression {

  // aligned to avoid false sharing between the counters of different streams
  struct alignas(std::hardware_destructive_interference_size) Prescaler::Counter {
    unsigned int value = 0;
    bool initialised = false;
  };

  Prescaler::Prescaler(Evaluator* arg, unsigned int prescale) : UnaryOperator(arg), m_prescale(prescale) {}

  Prescaler::~Prescaler() = default;

  bool Prescaler::operator()(const Data& data) const {
    // if the prescale factor is 0, we never need to run any dependent module,
    // so we can safely skip the rest of the processing
    if (m_prescale == 0)
      return false;

    // get the counter before evaluating the argument, so that it is initialised to the number of the first event
    // seen by this stream even if the argument is false
    unsigned int& counter = this->counter(data);

    bool result = ((*m_arg)(data));
    if (not result)
      return false;

    // if the prescale factor is 1, we do not need to keep track of the event counter
    if (m_prescale == 1)
      return true;

    return (++counter % m_prescale) == 0;
  }

  unsigned int& Prescaler::counter(const Data& data) const {
    // m_counters is sized by init() before the expression is shared between streams; here each stream only
    // accesses its own element, without resizing the vector, so that the streams can evaluate the expression
    // concurrently
    assert(data.streamIndex() < m_counters.size());
    Counter& counter = m_counters[data.streamIndex()];
    if (not counter.initialised) {
      counter.value = data.eventNumber();
      counter.initialised = true;
    }
    return counter.value;
  }

  void Prescaler::init(const Data& data) {
    // initialize the depending modules
    UnaryOperator::init(data);

    // one counter for each stream that shares this expression; init() is not called concurrently with the
    // evaluation of the expression
    m_counters.resize(data.numberOfStreams());

    // initialize the counter to the first event number seen,
    // in order to avoid all prescalers on different FUs to be synchronous
    counter(data) = data.eventNumber();
  }

}  // namespace triggerExpression
