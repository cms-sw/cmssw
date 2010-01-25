/** \class TriggerResultsFilter
 *
 * See header file for documentation
 *
 *  $Date: 2010/01/22 01:04:16 $
 *  $Revision: 1.8 $
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
  m_expression(0),
  m_eventCache(config)
{
  // parse the logical expressions into functionals
  std::string expression( config.getParameter<std::string>("triggerConditions") );
  m_expression = triggerExpression::parse( expression );

  // check if the expressions were parsed correctly
  if (not m_expression)
    edm::LogWarning("Configuration") << "Couldn't parse trigger results expression \"" << expression << "\"" << std::endl;
}

TriggerResultsFilter::~TriggerResultsFilter()
{
  delete m_expression;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool TriggerResultsFilter::filter(edm::Event & event, const edm::EventSetup & setup)
{
  if (not m_eventCache.setEvent(event, setup))
    // couldn't properly access all information from the Event
    return false;

  bool result = false;

  if (m_expression) {
    // run the trigger results filter
    result = (*m_expression)(m_eventCache);
 
    // if the L1 or HLT configurations have changed, log the expanded configuration
    // this must be done *after* running the Evaluator, as that triggers the update 
    if (m_eventCache.configurationUpdated())
      edm::LogInfo("Configuration") << "TriggerResultsFilter configuration updated: " << *m_expression;
  }

  return result; 
}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerResultsFilter);
