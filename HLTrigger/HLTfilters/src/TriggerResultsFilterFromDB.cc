/** \class TriggerResultsFilterFromDB
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/21 14:57:00 $
 *  $Revision: 1.2 $
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
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "HLTrigger/HLTfilters/interface/TriggerResultsFilterFromDB.h"

//
// constructors and destructor
//
TriggerResultsFilterFromDB::TriggerResultsFilterFromDB(const edm::ParameterSet & config) : HLTFilter(config),
  m_eventSetupPathsKey(config.getParameter<std::string>("eventSetupPathsKey")),
  m_eventSetupWatcher(),
  m_expression(0),
  m_eventCache(config)
{
}

TriggerResultsFilterFromDB::~TriggerResultsFilterFromDB()
{
  delete m_expression;
}

void TriggerResultsFilterFromDB::parse(const std::vector<std::string> & expressions) {
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

void TriggerResultsFilterFromDB::parse(const std::string & expression) {
  // parse the logical expressions into functionals
  m_expression = triggerExpression::parse( expression );

  // check if the expressions were parsed correctly
  if (not m_expression)
    edm::LogWarning("Configuration") << "Couldn't parse trigger results expression \"" << expression << "\"";
}

// read the triggerConditions from the database
void TriggerResultsFilterFromDB::pathsFromSetup(const edm::EventSetup & setup)
{
  // Get map of strings to concatenated list of names of HLT paths from EventSetup:
  edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
  setup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
  typedef std::map<std::string, std::string> TriggerMap;
  const TriggerMap & triggerMap = triggerBits->m_alcarecoToTrig;

  TriggerMap::const_iterator listIter = triggerMap.find(m_eventSetupPathsKey);
  if (listIter == triggerMap.end()) {
    throw cms::Exception("Configuration") << "TriggerResultsFilterFromDB [instance: " << * moduleLabel() 
                                          << " - path: " << * pathName() 
                                          << "]: No triggerList with key " << m_eventSetupPathsKey << " in AlCaRecoTriggerBitsRcd";
  }

  // avoid a map<string,vector<string> > in DB for performance reason,
  // the paths are mapped into one string that we have to decompose:
  parse( triggerBits->decompose(listIter->second) );
}

bool TriggerResultsFilterFromDB::hltFilter(edm::Event & event, const edm::EventSetup & setup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  // if the IOV has changed, re-read the triggerConditions from the database
  if (m_eventSetupWatcher.check(setup))
    pathsFromSetup(setup);

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
    edm::LogInfo("Configuration") << "TriggerResultsFilterFromDB configuration updated: " << *m_expression;
  }

  // run the trigger results filter
  return (*m_expression)(m_eventCache);
}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerResultsFilterFromDB);
