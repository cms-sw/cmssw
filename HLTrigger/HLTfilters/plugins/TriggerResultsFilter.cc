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
#include <algorithm>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/Utilities/interface/transform.h"
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
    // if the expression was successfully parsed, store the list of patterns,
    // each in both std::string and std::regex format
    // and with a flag for having valid matches
    hltPathStatusPatterns_ = edm::vector_transform(m_expression->patterns(), [](std::string const& pattern) {
      return PatternData(pattern, std::regex(edm::glob2reg(pattern), std::regex::extended));
    });

    if (hltPathStatusPatterns_.empty()) {
      return;
    }

    // consume all matching paths
    callWhenNewProductsRegistered([this](edm::BranchDescription const& branch) {
      if (branch.branchType() == edm::InEvent and branch.className() == "edm::HLTPathStatus") {
        bool consumeBranch = true;
        for (auto& pattern : hltPathStatusPatterns_) {
          if (std::regex_match(branch.moduleLabel(), pattern.regex)) {
            pattern.matched = true;
            if (consumeBranch) {
              consumeBranch = false;
              m_eventCache.setPathStatusToken(branch, consumesCollector());
            }
          }
        }
      }
    });
  }
}

void TriggerResultsFilter::beginStream(edm::StreamID) {
  // if throw=True, check if any of the input patterns had zero matches (and if so, throw an exception)
  if (not hltPathStatusPatterns_.empty() and m_eventCache.shouldThrow()) {
    auto unmatchedPatternsExist = std::any_of(
        hltPathStatusPatterns_.cbegin(), hltPathStatusPatterns_.cend(), [](auto foo) { return (not foo.matched); });
    if (unmatchedPatternsExist) {
      cms::Exception excpt("UnmatchedPatterns");
      excpt << "the parameter \"triggerConditions\" contains patterns with zero matches"
            << " for the available edm::HLTPathStatus collections - invalid patterns are:";
      for (auto const& pattern : hltPathStatusPatterns_)
        if (not pattern.matched)
          excpt << "\n\t" << pattern.str;
      throw excpt;
    }
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
  if (not m_expression) {
    if (m_eventCache.shouldThrow()) {
      throw cms::Exception("Configuration") << "Couldn't parse trigger-results expression \"" << expression << "\"";
    } else {
      edm::LogWarning("Configuration") << "Couldn't parse trigger-results expression \"" << expression << "\"";
    }
  }
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
