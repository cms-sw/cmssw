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
#include <string>
#include <vector>
#include <regex>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

// forward declaration
namespace edm {
  class ConfigurationDescriptions;
}
namespace triggerExpression {
  class Evaluator;
}

//
// class declaration
//

class TriggerResultsFilter : public edm::stream::EDFilter<> {
public:
  explicit TriggerResultsFilter(const edm::ParameterSet &);
  ~TriggerResultsFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  bool filter(edm::Event &, const edm::EventSetup &) override;

private:
  void beginStream(edm::StreamID) override;

  /// parse the logical expression into functionals
  void parse(const std::string &expression);
  void parse(const std::vector<std::string> &expressions);

  /// evaluator for the trigger condition
  std::unique_ptr<triggerExpression::Evaluator> m_expression;

  /// cache some data from the Event for faster access by the m_expression
  triggerExpression::Data m_eventCache;

  struct PatternData {
    PatternData(std::string const &aStr, std::regex const &aRegex, bool const hasMatch = false)
        : str(aStr), regex(aRegex), matched(hasMatch) {}
    std::string str;
    std::regex regex;
    bool matched;
  };

  std::vector<PatternData> hltPathStatusPatterns_;
};

#endif  //TriggerResultsFilter_h
