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
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

namespace triggerExpression {

template <typename T>
static 
const T * get(const edm::Event & event, const edm::EDGetTokenT<T> & token) {
  edm::Handle<T> handle;
  event.getByToken(token, handle);
  if (not handle.isValid()) {
    auto const & error = handle.whyFailed();
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
  if (hasL1T()) {
    // cache the L1 GT results objects
    m_l1tResults = get<L1GlobalTriggerReadoutRecord>(event, m_l1tResultsToken);
    if (not m_l1tResults)
      return false;

    // cache the L1 trigger masks
    m_l1tAlgoMask = get<L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask>(setup);
    m_l1tTechMask = get<L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask>(setup);

    // cache the L1 trigger menu
    unsigned long long l1tCacheID = setup.get<L1GtTriggerMenuRcd>().cacheIdentifier();
    if (m_l1tCacheID == l1tCacheID) {
      m_l1tUpdated = false;
    } else {
      m_l1tMenu = get<L1GtTriggerMenuRcd, L1GtTriggerMenu>(setup);
      m_l1tCacheID = l1tCacheID;
      m_l1tUpdated = true;
    }
  }

  // access HLT objects only if HLT is used
  if (hasHLT()) {
    // cache the HLT TriggerResults
    m_hltResults = get<edm::TriggerResults>(event, m_hltResultsToken);
    if (not m_hltResults)
      return false;

    // access the TriggerNames, and check if it has changed
    m_hltMenu = & event.triggerNames(* m_hltResults);
    if (m_hltMenu->parameterSetID() == m_hltCacheID) {
      m_hltUpdated = false;
    } else {
      m_hltCacheID = m_hltMenu->parameterSetID();
      m_hltUpdated = true;
    }
  }

  return true;
}

} // namespace triggerExpression
