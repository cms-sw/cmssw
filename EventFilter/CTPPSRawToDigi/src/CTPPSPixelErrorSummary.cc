#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelErrorSummary.h"
#include <iostream>
#include <algorithm>

void CTPPSPixelErrorSummary::add(const std::string& message, const std::string& details) {
  const auto eIt = m_errors.find(message);
  if (eIt == m_errors.end()) {
    m_errors.emplace(message, 1);
    edm::LogError(m_category) << message << ": " << details
                                << (m_debug ? ""
                                            : "\nNote: further warnings of this type will be suppressed (this can be "
                                              "changed by enabling debugging printout)");
  } else {
    ++(eIt->second);
    if (m_debug) {
      edm::LogError(m_category) << message << ": " << details;
    }
  }
}

void CTPPSPixelErrorSummary::printSummary() const {
  if (!m_errors.empty()) {
    std::stringstream message;
    message << m_name << " errors:";
    for (const auto& warnAndCount : m_errors) {
      message << std::endl << warnAndCount.first << " (" << warnAndCount.second << ")";
    }
    edm::LogError(m_category) << message.str();
  }
}
