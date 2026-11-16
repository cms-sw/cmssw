#ifndef TriggerResultsFilter_h
#define TriggerResultsFilter_h

/** \class TriggerResultsFilter
 *
 *
 *  This class is an EDFilter implementing filtering on arbitrary logical combinations
 *  of L1 and HLT results.
 *
 *  It has been written as an extension of the HLTHighLevel and HLTHighLevelDev filters.
 *
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

// forward declaration
namespace edm {
  class ConfigurationDescriptions;
}
namespace triggerExpression {
  class Evaluator;
}

namespace triggerResultsFilter {
  // per-stream state: the event data, and the expression initialised for the current L1 and HLT configuration
  struct StreamCache {
    explicit StreamCache(triggerExpression::Data const &data) : data(data) {}

    triggerExpression::Data data;
    std::shared_ptr<triggerExpression::Evaluator const> expression;
  };
}  // namespace triggerResultsFilter

//
// class declaration
//

class TriggerResultsFilter : public edm::global::EDFilter<edm::StreamCache<triggerResultsFilter::StreamCache>> {
public:
  explicit TriggerResultsFilter(const edm::ParameterSet &);
  ~TriggerResultsFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  void preallocate(edm::PreallocationConfiguration const &) override;
  void beginJob() override;
  std::unique_ptr<triggerResultsFilter::StreamCache> beginStream(edm::StreamID) const override;

  /// combine the logical expressions into a single one
  static std::string combine(const std::vector<std::string> &expressions);

  /// parse the logical expression into functionals
  std::unique_ptr<triggerExpression::Evaluator> parse() const;

  /// return the expression initialised for the L1 and HLT configuration of the given event data
  std::shared_ptr<triggerExpression::Evaluator const> expressionFor(triggerExpression::Data const &data) const;

  /// the logical expression
  std::string m_expressionString;

  /// configuration and tokens, copied into the per-stream event data
  triggerExpression::Data m_eventCache;

  /// the L1 and HLT configuration for which an expression has been initialised
  struct ConfigurationKey {
    explicit ConfigurationKey(triggerExpression::Data const &data)
        : hltCacheID(data.m_hltCacheID), l1tCacheID(data.m_l1tCacheID), triggerNames(data.m_triggerNames) {}

    bool operator==(ConfigurationKey const &other) const = default;

    edm::ParameterSetID hltCacheID;
    unsigned long long l1tCacheID;
    std::vector<std::string> triggerNames;
  };

  /// protects m_parsedExpression and m_expressions
  mutable std::mutex m_mutex;

  /// expression parsed in the constructor, used for the first configuration
  mutable std::unique_ptr<triggerExpression::Evaluator> m_parsedExpression;

  /// expressions initialised so far, one for each configuration, shared by all streams
  mutable std::vector<std::pair<ConfigurationKey, std::shared_ptr<triggerExpression::Evaluator const>>> m_expressions;

  /// whether the expression could be parsed
  bool m_valid = false;

  class PatternData {
  public:
    explicit PatternData(std::string const &pattern);
    // check if the pattern matches the given name, and record a positive match
    bool matches(std::string const &name);

    std::string const &str() const { return str_; }
    bool matched() const { return matched_; }

  private:
    std::string str_;
    std::optional<std::regex> regex_;
    bool matched_ = false;
  };

  std::vector<PatternData> hltPathStatusPatterns_;
};

#endif  //TriggerResultsFilter_h
