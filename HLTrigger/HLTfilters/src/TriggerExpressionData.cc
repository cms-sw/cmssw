#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionData.h"

namespace triggerExpression {

template <typename T>
static 
const T * get(const edm::Event & event, const edm::InputTag & tag) {
  edm::Handle<T> handle;
  event.getByLabel(tag, handle);
  if (not handle.isValid()) {
    boost::shared_ptr<cms::Exception> const & error = handle.whyFailed();
    edm::LogWarning(error->category()) << error->what();
    return 0;
  } else {
    return handle.product();
  }
}

template <typename R, typename T>
static
const T * get(const edm::EventSetup & setup) {
  edm::ESHandle<T> handle;
  setup.get<R>().get(handle);
  return handle.product();
}

bool Data::setEvent(const edm::Event & event, const edm::EventSetup & setup) {

  // cache the event number
  m_eventNumber = event.id().event();

  // access L1 objects only if L1 is used
  if (not m_l1tResultsTag.label().empty()) {
    // cache the L1 GT results objects
    m_l1tResults = get<L1GlobalTriggerReadoutRecord>(event, m_l1tResultsTag);
    if (not m_l1tResults)
      return false;

    // cache the L1 masks
    m_l1tAlgoMask = get<L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask>(setup);
    m_l1tTechMask = get<L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask>(setup);
  }

  // access HLT objects only if HLT is used
  if (not m_hltResultsTag.label().empty()) {
    // cache the HLT TriggerResults
    m_hltResults = get<edm::TriggerResults>(event, m_hltResultsTag);
    if (not m_hltResults)
      return false;

    // access the TriggerNames, and check if it has changed
    m_hltNames = & event.triggerNames(* m_hltResults);
    if (m_hltNames->parameterSetID() == m_hltPSetID) {
      m_hltUpdated = false;
    } else {
      m_hltPSetID  = m_hltNames->parameterSetID();
      m_hltUpdated = true;
    }
  }

  return true;
}

} // namespace triggerExpression
