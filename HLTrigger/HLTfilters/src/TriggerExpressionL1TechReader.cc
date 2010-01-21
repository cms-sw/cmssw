#include <iostream>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionL1TechReader.h"

namespace triggerExpression {

// define the result of the module from the L1 reults
bool L1TechReader::operator()(const Data & data) {
  return false;
}

void L1TechReader::dump(std::ostream & out) const {
  out << "FALSE";
}

void L1TechReader::init(const Data & data) {
}

} // namespace triggerExpression

