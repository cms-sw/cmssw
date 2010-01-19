#include "HLTrigger/HLTfilters/interface/TriggerExpressionL1Reader.h"

namespace hlt {

// define the result of the module from the L1 reults
bool TriggerExpressionL1Reader::operator()(const TriggerExpressionCache & data) {
  return false;
}

void TriggerExpressionHLTReader::dump(std::ostream & out) const {
  out << "FALSE";
}

} // namespace hlt
