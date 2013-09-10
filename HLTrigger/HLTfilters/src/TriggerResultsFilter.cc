/** \class TriggerResultsFilter
 *
 * See header file for documentation
 *
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <boost/foreach.hpp>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "HLTrigger/HLTfilters/interface/TriggerResultsFilter.h"

//
// constructors and destructor
//
TriggerResultsFilter::TriggerResultsFilter(const edm::ParameterSet & config) :
  m_expression(0),
  m_eventCache(config, consumesCollector())
{
  const std::vector<std::string> & expressions = config.getParameter<std::vector<std::string> >("triggerConditions");
  parse( expressions );
}

TriggerResultsFilter::~TriggerResultsFilter()
{
  delete m_expression;
}

void
TriggerResultsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // # HLT results   - set to empty to ignore HLT
  desc.add<edm::InputTag>("hltResults",edm::InputTag("TriggerResults"));
  // # L1 GT results - set to empty to ignore L1T
  desc.add<edm::InputTag>("l1tResults",edm::InputTag("hltGtDigis"));
  // # use L1 mask
  desc.add<bool>("l1tIgnoreMask",false);
  // # read L1 technical bits from PSB#9, bypassing the prescales
  desc.add<bool>("l1techIgnorePrescales",false);
  // # used by the definition of the L1 mask
  desc.add<unsigned int>("daqPartitions",0x01);
  // # throw exception on unknown trigger names
  desc.add<bool>("throw",true);
  // # trigger conditions
  std::vector<std::string> triggerConditions(1,"HLT_*");
  desc.add<std::vector<std::string> >("triggerConditions",triggerConditions);
  descriptions.add("triggerResultsFilter", desc);
}

void TriggerResultsFilter::parse(const std::vector<std::string> & expressions) {
  // parse the logical expressions into functionals
  if (expressions.size() == 0) {
    edm::LogWarning("Configuration") << "Empty trigger results expression";
  } else if (expressions.size() == 1) {
    parse( expressions[0] );
  } else {
    std::stringstream expression;
    expression << "(" << expressions[0] << ")";
    for (unsigned int i = 1; i < expressions.size(); ++i)
      expression << " OR (" << expressions[i] << ")";
    parse( expression.str() );
  }
}

void TriggerResultsFilter::parse(const std::string & expression) {
  // parse the logical expressions into functionals
  m_expression = triggerExpression::parse( expression );

  // check if the expressions were parsed correctly
  if (not m_expression)
    edm::LogWarning("Configuration") << "Couldn't parse trigger results expression \"" << expression << "\"";
}

bool TriggerResultsFilter::filter(edm::Event & event, const edm::EventSetup & setup)
{
  if (not m_expression)
    // no valid expression has been parsed
    return false;

  if (not m_eventCache.setEvent(event, setup))
    // couldn't properly access all information from the Event
    return false;

  // if the L1 or HLT configurations have changed, (re)initialize the filters (including during the first event)
  if (m_eventCache.configurationUpdated()) {
    m_expression->init(m_eventCache);

    // log the expanded configuration
    edm::LogInfo("Configuration") << "TriggerResultsFilter configuration updated: " << *m_expression;
  }

  // run the trigger results filter
  return (*m_expression)(m_eventCache);
}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerResultsFilter);
