/** \class TriggerResultsFilter
 *
 * See header file for documentation
 *
 *  $Date: 2010/01/22 00:19:42 $
 *  $Revision: 1.7 $
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <boost/version.hpp>
#if BOOST_VERSION < 104100
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <boost/foreach.hpp>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTfilters/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionParser.h"
#include "HLTrigger/HLTfilters/interface/TriggerResultsFilter.h"

//
// constructors and destructor
//
TriggerResultsFilter::TriggerResultsFilter(const edm::ParameterSet & config) :
  m_expressions(),
  m_eventCache(config)
{
  // parse the logical expressions into functionals
  const std::vector<std::string> & expressions = config.getParameter<std::vector<std::string> >("triggerConditions");
  unsigned int size = expressions.size();
  m_expressions.resize(size);
  for (unsigned int i = 0; i < size; ++i) {
    m_expressions[i] = triggerExpression::parse(expressions[i]);
    // check if the expressions were parsed correctly
    if (not m_expressions[i])
      edm::LogWarning("Configuration") << "Couldn't parse trigger results expression \"" << expressions[i] << "\"" << std::endl;
  }
}

TriggerResultsFilter::~TriggerResultsFilter()
{
  for (unsigned int i = 0; i < m_expressions.size(); ++i)
    delete m_expressions[i];
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool TriggerResultsFilter::filter(edm::Event & event, const edm::EventSetup & setup)
{
  if (not m_eventCache.setEvent(event, setup)) {
    // couldn't properly access all information from the Event
    return false;
  }

  // run the trigger results filters
  bool result = false;
  BOOST_FOREACH(triggerExpression::Evaluator * expression, m_expressions)
    if (expression and (*expression)(m_eventCache))
      result = true;
 
  // if the L1 or HLT configurations have changed, log the expanded configuration
  // this must be done *after* running the Evaluator, as that triggers the update 
  if (m_eventCache.configurationUpdated()) {
    std::stringstream out;
    out << "TriggerResultsFilter configuration updated:";
    BOOST_FOREACH(triggerExpression::Evaluator * expression, m_expressions)
      if (expression) 
        out << "\n\t" << (*expression);
      else
        out << "\n\tFALSE";
    edm::LogInfo("Configuration") << out.str();
  }

  return result; 
}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerResultsFilter);
