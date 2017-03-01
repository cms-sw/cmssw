#ifndef TriggerResultsFilterFromDB_h
#define TriggerResultsFilterFromDB_h

/** \class TriggerResultsFilterFromDB
 *
 *
 *  This class is an EDFilter implementing filtering on arbitrary logical combinations
 *  of L1 and HLT results.
 *
 *  It is a modifed version of TriggerResultsFilter that reads the trigger expression
 *  from the database.
 *
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <vector>
#include <string>

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"
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

class TriggerResultsFilterFromDB : public edm::stream::EDFilter<>
{
public:
  explicit TriggerResultsFilterFromDB(const edm::ParameterSet &);
  ~TriggerResultsFilterFromDB();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  bool filter(edm::Event &, const edm::EventSetup &) override;

private:
  /// read the triggerConditions from the database
  void pathsFromSetup(const edm::Event &, const edm::EventSetup & setup);

  /// parse the logical expression into functionals
  void parse(const std::string & expression);
  void parse(const std::vector<std::string> & expressions);

  /// read the triggerConditions from the database
  std::string m_eventSetupPathsKey;
  edm::ESWatcher<AlCaRecoTriggerBitsRcd> m_eventSetupWatcher;

  /// evaluator for the trigger condition
  triggerExpression::Evaluator * m_expression;

  /// cache some data from the Event for faster access by the m_expression
  triggerExpression::Data m_eventCache;
};

#endif //TriggerResultsFilterFromDB_h
