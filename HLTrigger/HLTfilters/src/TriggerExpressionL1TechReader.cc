#include "HLTrigger/HLTfilters/interface/TriggerExpressionL1TechReader.h"

namespace triggerExpression {

// define the result of the module from the L1 reults
bool L1TechReader::operator()(const Data & data) {
  return false;
}

void L1TechReader::dump(std::ostream & out) const {
  out << "FALSE";
}

} // namespace triggerExpression

