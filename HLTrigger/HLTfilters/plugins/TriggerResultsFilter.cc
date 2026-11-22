/** \class TriggerResultsFilter
 *
 * See header file for documentation
 *
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
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
// path name patterns
//
TriggerResultsFilter::PatternData::PatternData(std::string const& pattern) : str_(pattern) {
  // a pattern made only of letters, digits and underscores matches only itself, so it is compared as a string;
  // a std::regex is built only for the other patterns
  bool plain = std::all_of(pattern.begin(), pattern.end(), [](unsigned char c) { return std::isalnum(c) or c == '_'; });
  if (not plain) {
    regex_.emplace(edm::glob2reg(pattern), std::regex::extended);
  }
}

bool TriggerResultsFilter::PatternData::matches(std::string const& name) {
  bool match = regex_ ? std::regex_match(name, *regex_) : name == str_;
  if (match) {
    matched_ = true;
  }
  return match;
}

//
// constructors and destructor
//
TriggerResultsFilter::TriggerResultsFilter(const edm::ParameterSet& config)
    : m_expressionString(combine(config.getParameter<std::vector<std::string>>("triggerConditions"))),
      m_eventCache(config, consumesCollector()) {
  m_parsedExpression = parse();
  m_valid = static_cast<bool>(m_parsedExpression);
  if (m_valid and m_eventCache.usePathStatus()) {
    // if the expression was successfully parsed, store the list of patterns,
    // each as a std::string and, unless it is a plain name, as a std::regex,
    // and with a flag for having valid matches
    hltPathStatusPatterns_ = edm::vector_transform(m_parsedExpression->patterns(),
                                                   [](std::string const& pattern) { return PatternData(pattern); });

    if (hltPathStatusPatterns_.empty()) {
      return;
    }

    // consume all matching paths
    callWhenNewProductsRegistered([this](edm::ProductDescription const& branch) {
      if (branch.branchType() == edm::InEvent and branch.className() == "edm::HLTPathStatus") {
        bool consumeBranch = false;
        for (auto& pattern : hltPathStatusPatterns_) {
          // test every pattern, to record all the patterns that match this path
          if (pattern.matches(branch.moduleLabel()))
            consumeBranch = true;
        }
        if (consumeBranch)
          m_eventCache.setPathStatusToken(branch, consumesCollector());
      }
    });
  }
}

void TriggerResultsFilter::preallocate(edm::PreallocationConfiguration const& config) {
  // the expressions are shared by all streams, and need one prescale counter per stream
  m_eventCache.setNumberOfStreams(config.numberOfStreams());
}

void TriggerResultsFilter::beginJob() {
  // if throw=True, check if any of the input patterns had zero matches (and if so, throw an exception)
  if (not hltPathStatusPatterns_.empty() and m_eventCache.shouldThrow()) {
    auto unmatchedPatternsExist = std::any_of(hltPathStatusPatterns_.cbegin(),
                                              hltPathStatusPatterns_.cend(),
                                              [](auto const& foo) { return (not foo.matched()); });
    if (unmatchedPatternsExist) {
      cms::Exception excpt("UnmatchedPatterns");
      excpt << "the parameter \"triggerConditions\" contains patterns with zero matches"
            << " for the available edm::HLTPathStatus collections - invalid patterns are:";
      for (auto const& pattern : hltPathStatusPatterns_)
        if (not pattern.matched())
          excpt << "\n\t" << pattern.str();
      throw excpt;
    }
  }
  // the patterns are no longer needed
  hltPathStatusPatterns_.clear();
  hltPathStatusPatterns_.shrink_to_fit();
}

std::unique_ptr<triggerResultsFilter::StreamCache> TriggerResultsFilter::beginStream(edm::StreamID) const {
  // copy the configuration and the tokens; the event data is filled by setEvent()
  return std::make_unique<triggerResultsFilter::StreamCache>(m_eventCache);
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

std::string TriggerResultsFilter::combine(const std::vector<std::string>& expressions) {
  // combine the logical expressions with OR
  if (expressions.empty()) {
    edm::LogWarning("Configuration") << "Empty trigger results expression";
    return {};
  }
  if (expressions.size() == 1) {
    return expressions[0];
  }
  std::stringstream expression;
  expression << "(" << expressions[0] << ")";
  for (unsigned int i = 1; i < expressions.size(); ++i)
    expression << " OR (" << expressions[i] << ")";
  return expression.str();
}

std::unique_ptr<triggerExpression::Evaluator> TriggerResultsFilter::parse() const {
  if (m_expressionString.empty()) {
    return {};
  }
  // parse the logical expressions into functionals
  std::unique_ptr<triggerExpression::Evaluator> expression(triggerExpression::parse(m_expressionString));
  // check if the expressions were parsed correctly
  if (not expression) {
    if (m_eventCache.shouldThrow()) {
      throw cms::Exception("Configuration")
          << "Couldn't parse trigger-results expression \"" << m_expressionString << "\"";
    } else {
      edm::LogWarning("Configuration") << "Couldn't parse trigger-results expression \"" << m_expressionString << "\"";
    }
  }
  return expression;
}

std::shared_ptr<triggerExpression::Evaluator const> TriggerResultsFilter::expressionFor(
    triggerExpression::Data const& data) const {
  ConfigurationKey key(data);
  std::lock_guard<std::mutex> guard(m_mutex);
  for (auto const& [k, expression] : m_expressions) {
    if (k == key) {
      return expression;
    }
  }
  // initialise an expression for this configuration: the one parsed in the constructor the first time, a newly
  // parsed one afterwards; the expressions already used by other streams are not modified, as they may still be
  // in use for events with a previous configuration
  std::unique_ptr<triggerExpression::Evaluator> expression =
      m_parsedExpression ? std::move(m_parsedExpression) : parse();
  expression->init(data);
  // log the expanded configuration
  edm::LogInfo("Configuration") << "TriggerResultsFilter configuration updated: " << *expression;
  std::shared_ptr<triggerExpression::Evaluator const> shared = std::move(expression);
  m_expressions.emplace_back(std::move(key), shared);
  return shared;
}

bool TriggerResultsFilter::filter(edm::StreamID sid, edm::Event& event, const edm::EventSetup& setup) const {
  if (not m_valid)
    // no valid expression has been parsed
    return false;

  auto& cache = *streamCache(sid);
  if (not cache.data.setEvent(event, setup))
    // couldn't properly access all information from the Event
    return false;

  // if the L1 or HLT configurations have changed (including during the first event),
  // use the expression initialised for the new configuration
  if (cache.data.configurationUpdated() or not cache.expression) {
    cache.expression = expressionFor(cache.data);
  }

  // run the trigger results filter
  return (*cache.expression)(cache.data);
}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerResultsFilter);
