#include "HLTrigger/HLTfilters/interface/TriggerExpressionL1Reader.h"

namespace triggerExpression {

// define the result of the module from the L1 reults
bool L1Reader::operator()(const Data & data) {
  return false;
}

void L1Reader::dump(std::ostream & out) const {
  out << "FALSE";
}

} // namespace triggerExpression
