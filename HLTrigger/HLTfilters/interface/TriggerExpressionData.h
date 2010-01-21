#ifndef HLTrigger_HLTfilters_TriggerExpressionData_h
#define HLTrigger_HLTfilters_TriggerExpressionData_h

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/EventID.h"

namespace edm {
  class Event;
  class EventSetup;
  class TriggerResults;
  class TriggerNames;
}

class L1GlobalTriggerReadoutRecord;
class L1GtTriggerMask;

namespace triggerExpression {

class Data {
public:
  explicit Data(const edm::ParameterSet & config) :
    // configuration
    m_hltResultsTag(config.getParameter<edm::InputTag>("hltResults")),
    m_l1tResultsTag(config.getParameter<edm::InputTag>("l1tResults")),
    m_l1tIgnoreMask(config.getParameter<bool>("l1tIgnoreMask")),
    m_throw(config.getParameter<bool>("throw")),
    // values
    m_hltResults(0),
    m_hltNames(0),
    m_hltPSetID(),
    m_hltUpdated(false),
    m_eventNumber()
  { }

  bool setEvent(const edm::Event & event, const edm::EventSetup & setup);

  const edm::TriggerResults & triggerResults() const {
    return * m_hltResults;
  }

  const edm::TriggerNames & triggerNames() const {
    return * m_hltNames;
  }

  bool configurationUpdated() const {
    return m_hltUpdated;
  }

  edm::EventNumber_t eventNumber() const {
    return m_eventNumber;
  }

private:
  // configuration
  edm::InputTag m_hltResultsTag;
  edm::InputTag m_l1tResultsTag;
  bool m_l1tIgnoreMask;
  bool m_throw;

  // values
  const L1GlobalTriggerReadoutRecord * m_l1tResults;
  const L1GtTriggerMask     * m_l1tAlgoMask;
  const L1GtTriggerMask     * m_l1tTechMask;
  const edm::TriggerResults * m_hltResults;
  const edm::TriggerNames   * m_hltNames;

  edm::ParameterSetID         m_hltPSetID;
  bool                        m_hltUpdated;

  edm::EventNumber_t          m_eventNumber;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionData_h
