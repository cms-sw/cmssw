#include "HLTrigger/HLTfilters/interface/TriggerExpressionL1TechReader.h"

namespace hlt {

// define the result of the module from the L1 reults
bool TriggerExpressionL1TechReader::operator()(const TriggerExpressionData & data) {
  return false;
}

void TriggerExpressionL1TechReader::dump(std::ostream & out) const {
  out << "FALSE";
}

} // namespace hlt

