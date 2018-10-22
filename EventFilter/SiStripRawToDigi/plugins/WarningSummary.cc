#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/SiStripRawToDigi/plugins/WarningSummary.h"

#include <algorithm>

void sistrip::WarningSummary::add(const std::string& message, const std::string& details)
{
  const auto wIt = std::find_if(std::begin(m_warnings), std::end(m_warnings),
      [&message] ( const std::pair<std::string,std::size_t>& item ) { return item.first == message; } );
  if ( std::end(m_warnings) == wIt ) {
    m_warnings.emplace(message, 1);
    edm::LogWarning(m_category) << message << ": " << details << (m_debug?"":
        "\nNote: further warnings of this type will be suppressed (this can be changed by enabling debugging printout)");
  } else {
    ++(wIt->second);
    if ( m_debug ) {
      edm::LogWarning(m_category) << message << ": " << details;
    }
  }
}

void sistrip::WarningSummary::printSummary() const
{
  if ( ! m_warnings.empty() ) {
    std::stringstream message;
    message << m_name << " warnings:";
    for ( const auto& warnAndCount : m_warnings ) {
      message << std::endl << warnAndCount.first << " (" << warnAndCount.second << ")";
    }
    edm::LogWarning(m_category) << message.str();
  }
}
