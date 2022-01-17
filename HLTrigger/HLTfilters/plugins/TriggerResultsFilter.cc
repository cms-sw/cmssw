/** \class TriggerResultsFilter
 *
 * See header file for documentation
 *
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"

#include "TriggerResultsFilter.h"

//
// constructors and destructor
//
TriggerResultsFilter::TriggerResultsFilter(const edm::ParameterSet& config)
    : m_expression(nullptr), m_eventCache(config, consumesCollector()) {
  std::vector<std::string> const& expressions = config.getParameter<std::vector<std::string>>("triggerConditions");
  parse(expressions);
  if (m_expression and m_eventCache.usePathStatus()) {
    // if the expression was succesfully parsed, join all the patterns corresponding
    // to the CMSSW paths in the logical expression into a single regex
    std::vector<std::string> patterns = m_expression->patterns();
    if (patterns.empty()) {
      return;
    }
    std::string str;
    for (auto const& pattern : patterns) {
      str += edm::glob2reg(pattern);
      str += '|';
    }
    str.pop_back();
    std::regex regex(str, std::regex::extended);

    // consume all matching paths
    callWhenNewProductsRegistered([this, regex](edm::BranchDescription const& branch) {
      if (branch.branchType() == edm::InEvent and branch.className() == "edm::HLTPathStatus" and
          std::regex_match(branch.moduleLabel(), regex)) {
        m_eventCache.setPathStatusToken(branch, consumesCollector());
      }
    });
  }
}

void TriggerResultsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // # use HLTPathStatus results
  desc.add<bool>("usePathStatus", false)
      ->setComment("Read the HLT results from the TriggerResults (false) or from the current job's PathStatus (true).");
  // # HLT results - set to empty to ignore HLT
  desc.add<edm::InputTag>("hltResults", edm::InputTag("TriggerResults", "", "@skipCurrentProcess"))
      ->setComment("HLT TriggerResults. Leave empty to ignore the HLT results. Ignored when usePathStatus is true.");
  // # L1 uGT results - set to empty to ignore L1T
  desc.add<edm::InputTag>("l1tResults", edm::InputTag("hltGtStage2Digis"))
      ->setComment("uGT digi collection. Leave empty to ignore the L1T results.");
  // # use initial L1 decision, before masks and prescales
  desc.add<bool>("l1tIgnoreMaskAndPrescale", false);
  // # OBSOLETE - these parameters are ignored, they are left only not to break old configurations
  // they will not be printed in the generated cfi.py file
  desc.addOptionalNode(edm::ParameterDescription<bool>("l1tIgnoreMask", false, true), false)
      ->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("l1techIgnorePrescales", false, true), false)
      ->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<unsigned int>("daqPartitions", 0x01, true), false)
      ->setComment("This parameter is obsolete and will be ignored.");
  // # throw exception on unknown trigger names
  desc.add<bool>("throw", true);
  // # trigger conditions
  std::vector<std::string> triggerConditions(1, "HLT_*");
  desc.add<std::vector<std::string>>("triggerConditions", triggerConditions);
  descriptions.add("triggerResultsFilter", desc);
}

void TriggerResultsFilter::parse(const std::vector<std::string>& expressions) {
  // parse the logical expressions into functionals
  if (expressions.empty()) {
    edm::LogWarning("Configuration") << "Empty trigger results expression";
  } else if (expressions.size() == 1) {
    parse(expressions[0]);
  } else {
    std::stringstream expression;
    expression << "(" << expressions[0] << ")";
    for (unsigned int i = 1; i < expressions.size(); ++i)
      expression << " OR (" << expressions[i] << ")";
    parse(expression.str());
  }
}

void TriggerResultsFilter::parse(const std::string& expression) {
  // parse the logical expressions into functionals
  m_expression.reset(triggerExpression::parse(expression));

  // check if the expressions were parsed correctly
  if (not m_expression)
    edm::LogWarning("Configuration") << "Couldn't parse trigger results expression \"" << expression << "\"";
}

bool TriggerResultsFilter::filter(edm::Event& event, const edm::EventSetup& setup) {
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
