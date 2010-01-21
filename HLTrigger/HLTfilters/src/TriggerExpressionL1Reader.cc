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
#include "HLTrigger/HLTfilters/interface/TriggerExpressionL1Reader.h"

namespace triggerExpression {

// define the result of the module from the L1 reults
bool L1Reader::operator()(const Data & data) {
  if (not data.hasL1T())
    return false;

  if (data.l1tConfigurationUpdated())
    init(data.l1tMenu(), data.l1tAlgoMask());

  const std::vector<bool> & result = data.l1tResults().decisionWord();
  typedef std::pair<std::string, unsigned int> value_type;
  BOOST_FOREACH(const value_type & trigger, m_triggers)
    if (result[trigger.second])
      return true;

  return false;
}

void L1Reader::dump(std::ostream & out) const {
  if (m_triggers.size() == 0) {
    out << "FALSE";
  } else if (m_triggers.size() == 1) {
    out << m_triggers[0].first;
  } else {
    out << "(" << m_triggers[0].first;
    for (unsigned int i = 1; i < m_triggers.size(); ++i)
      out << " OR " << m_triggers[i].first;
    out << ")";
  }
}

void L1Reader::init(const L1GtTriggerMenu & menu, const L1GtTriggerMask & mask) {
  m_triggers.clear();

  // check if the pattern has is a glob expression, or a single trigger name 
  if (not edm::is_glob(m_pattern)) {
    // no wildcard expression
    const AlgorithmMap & aliasMap = menu.gtAlgorithmAliasMap();
    AlgorithmMap::const_iterator entry = aliasMap.find(m_pattern);
    if (entry != aliasMap.end()) {
      // single L1 bit
      m_triggers.push_back( std::make_pair(m_pattern, entry->second.algoBitNumber()) );
    } else
      // trigger not found in the current menu
      if (m_throw)
        throw cms::Exception("Configuration") << "requested L1 trigger \"" << m_pattern << "\" does not exist in the current L1 menu";
      else
        edm::LogInfo("Configuration") << "requested L1 trigger \"" << m_pattern << "\" does not exist in the current L1 menu";
  } else {
    // expand wildcards in the pattern 
    boost::regex re(edm::glob2reg(m_pattern));
    const AlgorithmMap & aliasMap = menu.gtAlgorithmAliasMap();
    BOOST_FOREACH(const AlgorithmMap::value_type & entry, aliasMap)
      if (boost::regex_match(entry.first, re))
        m_triggers.push_back( std::make_pair(entry.first, entry.second.algoBitNumber()) );

    if (m_triggers.empty()) {
      // m_pattern does not match any L1 bits
      if (m_throw)
        throw cms::Exception("Configuration") << "requested pattern \"" << m_pattern <<  "\" does not match any L1 triggers in the current menu";
      else
        edm::LogInfo("Configuration") << "requested pattern \"" << m_pattern <<  "\" does not match any L1 triggers in the current menu";
    }
  }

}

} // namespace triggerExpression
