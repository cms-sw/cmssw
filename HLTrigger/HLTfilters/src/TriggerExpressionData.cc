#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionData.h"

namespace hlt {

bool TriggerExpressionData::setEvent(const edm::Event & event) {
  // cache the TriggerResults and TriggerNames objects
  edm::Handle<edm::TriggerResults> h_triggerResults;
  event.getByLabel(m_triggerResultsTag, h_triggerResults);
  if (not h_triggerResults.isValid()) {
    return false;
  }
  m_triggerResults = h_triggerResults.product();
  m_triggerNames   = & event.triggerNames(* m_triggerResults);

  // check if the TriggerNames has changed
  if (m_triggerNames->parameterSetID() == m_parameterSetID) {
    m_updated = false;
  } else {
    m_parameterSetID = m_triggerNames->parameterSetID();
    m_updated = true;
  }

  // cache the event number
  m_eventNumber = event.id().event();

  return true;
}

} // namespace hlt
