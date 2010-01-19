#include <boost/foreach.hpp>

#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionHLTReader.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionCache.h"

namespace hlt {

// define the result of the module from the HLT reults
bool TriggerExpressionHLTReader::operator()(const TriggerExpressionCache & data) {
  if (data.configurationUpdated())
    init(data.triggerNames());

  BOOST_FOREACH(int trigger, m_triggers)
    if (data.triggerResults().accept(trigger))
      return true;
  
  return false;
}

void TriggerExpressionHLTReader::dump(std::ostream & out) const {
  if (m_triggers.size() == 0) {
    out << "FALSE";
  } else if (m_triggers.size() == 1) {
    out << m_triggers[0];
  } else {
    out << "(" << m_triggers[0];
    for (unsigned int i = 1; i < m_triggers.size(); ++i)
      out << " OR " << m_triggers[i];
    out << ")";
  }
}

// (re)initialize the module
void TriggerExpressionHLTReader::init(const edm::TriggerNames & triggerNames) {
  m_triggers.clear();

  // check if the pattern has is a glob expression, or a single trigger name
  if (not edm::is_glob(m_pattern)) {
    // no wildcard expression
    unsigned int index = triggerNames.triggerIndex(m_pattern);
    if (index <= triggerNames.size())
      m_triggers.push_back(index);
    else
      if (m_throw)
        throw cms::Exception("Configuration") << "requested HLT path \"" << m_pattern << "\" does not exist";
      else
        edm::LogInfo("Configuration") << "requested HLT path \"" << m_pattern << "\" does not exist";
  } else {
    // expand wildcards in the pattern
    const std::vector< std::vector<std::string>::const_iterator > & matches = edm::regexMatch(triggerNames.triggerNames(), m_pattern);
    if (matches.empty()) {
      // m_pattern does not match any trigger paths
      if (m_throw)
        throw cms::Exception("Configuration") << "requested m_pattern \"" << m_pattern <<  "\" does not match any HLT paths";
      else
        edm::LogInfo("Configuration") << "requested m_pattern \"" << m_pattern <<  "\" does not match any HLT paths";
    } else {
      // store indices corresponding to the matching triggers
      BOOST_FOREACH(const std::vector<std::string>::const_iterator & match, matches)
        m_triggers.push_back( triggerNames.triggerIndex(*match) );
    }
  }
}

} // namespace hlt
