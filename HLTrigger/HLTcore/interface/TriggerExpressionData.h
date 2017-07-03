#ifndef HLTrigger_HLTfilters_TriggerExpressionData_h
#define HLTrigger_HLTfilters_TriggerExpressionData_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

namespace edm {
  class Event;
  class EventSetup;
  class TriggerNames;
}

class L1TUtmTriggerMenu;

namespace triggerExpression {

class Data {
public:
  // default c'tor
  Data() :
    // configuration
    m_hltResultsTag(""),
    m_hltResultsToken(),
    m_l1tResultsTag(""),
    m_l1tResultsToken(),
    m_l1tIgnoreMaskAndPrescale(false),
    m_throw(true),
    // l1 values and status
    m_l1tResults(nullptr),
    m_l1tMenu(nullptr),
    m_l1tCacheID(),
    m_l1tUpdated(false),
    // hlt values and status
    m_hltResults(nullptr),
    m_hltMenu(nullptr),
    m_hltCacheID(),
    m_hltUpdated(false),
    // event values
    m_eventNumber()
  { }

  // explicit c'tor from a ParameterSet
  explicit Data(const edm::ParameterSet & config, edm::ConsumesCollector && iC) :
    // configuration
    m_hltResultsTag(config.getParameter<edm::InputTag>("hltResults")),
    m_hltResultsToken(),
    m_l1tResultsTag(config.getParameter<edm::InputTag>("l1tResults")),
    m_l1tResultsToken(),
    m_l1tIgnoreMaskAndPrescale(config.getParameter<bool>("l1tIgnoreMaskAndPrescale")),
    m_throw(config.getParameter<bool>("throw")),
    // l1 values and status
    m_l1tResults(nullptr),
    m_l1tMenu(nullptr),
    m_l1tCacheID(),
    m_l1tUpdated(false),
    // hlt values and status
    m_hltResults(nullptr),
    m_hltMenu(nullptr),
    m_hltCacheID(),
    m_hltUpdated(false),
    // event values
    m_eventNumber()
      {
	if (not m_hltResultsTag.label().empty()) m_hltResultsToken = iC.consumes<edm::TriggerResults>(m_hltResultsTag);
	if (not m_l1tResultsTag.label().empty()) m_l1tResultsToken = iC.consumes<GlobalAlgBlkBxCollection>(m_l1tResultsTag);
      }

  // explicit c'tor from single arguments
  Data(
    edm::InputTag const & hltResultsTag,
    edm::InputTag const & l1tResultsTag,
    bool                  l1tIgnoreMaskAndPrescale,
    bool                  doThrow,
    edm::ConsumesCollector && iC  ) :
    // configuration
    m_hltResultsTag(hltResultsTag),
    m_hltResultsToken(),
    m_l1tResultsTag(l1tResultsTag),
    m_l1tResultsToken(),
    m_l1tIgnoreMaskAndPrescale(l1tIgnoreMaskAndPrescale),
    m_throw(doThrow),
    // l1 values and status
    m_l1tResults(nullptr),
    m_l1tMenu(nullptr),
    m_l1tCacheID(),
    m_l1tUpdated(false),
    // hlt values and status
    m_hltResults(nullptr),
    m_hltMenu(nullptr),
    m_hltCacheID(),
    m_hltUpdated(false),
    // event values
    m_eventNumber()
      {
	if (not m_hltResultsTag.label().empty()) m_hltResultsToken = iC.consumes<edm::TriggerResults>(m_hltResultsTag);
	if (not m_l1tResultsTag.label().empty()) m_l1tResultsToken = iC.consumes<GlobalAlgBlkBxCollection>(m_l1tResultsTag);
      }

  // set the new event
  bool setEvent(const edm::Event & event, const edm::EventSetup & setup);

  // re-configuration accessors 

  void setHltResultsTag(edm::InputTag const & tag) {
    m_hltResultsTag = tag;
  }

  void setL1tResultsTag(edm::InputTag const & tag) {
    m_l1tResultsTag = tag;
  }

  void setL1tIgnoreMaskAndPrescale(bool l1tIgnoreMaskAndPrescale) {
    m_l1tIgnoreMaskAndPrescale = l1tIgnoreMaskAndPrescale;
  }

  void setThrow(bool doThrow) {
    m_throw = doThrow;
  }

  // read-only accessors

  bool hasL1T() const {
    return not m_l1tResultsTag.label().empty();
  }

  bool hasHLT() const {
    return not m_hltResultsTag.label().empty();
  }

  const edm::TriggerResults & hltResults() const {
    return * m_hltResults;
  }

  const edm::TriggerNames & hltMenu() const {
    return * m_hltMenu;
  }

  const std::vector<bool> & l1tResults() const {
    return * m_l1tResults;
  }

  const L1TUtmTriggerMenu & l1tMenu() const {
    return * m_l1tMenu;
  }

  bool hltConfigurationUpdated() const {
    return m_hltUpdated;
  }

  bool l1tConfigurationUpdated() const {
    return m_l1tUpdated;
  }

  bool configurationUpdated() const {
    return m_hltUpdated or m_l1tUpdated;
  }

  edm::EventNumber_t eventNumber() const {
    return m_eventNumber;
  }

  bool shouldThrow() const {
    return m_throw;
  }

  bool ignoreL1MaskAndPrescale() const {
    return m_l1tIgnoreMaskAndPrescale;
  }

  // configuration
  edm::InputTag                                 m_hltResultsTag;
  edm::EDGetTokenT<edm::TriggerResults>         m_hltResultsToken;
  edm::InputTag                                 m_l1tResultsTag;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection>    m_l1tResultsToken;
  bool                                          m_l1tIgnoreMaskAndPrescale;
  bool                                          m_throw;

  // l1 values and status
  const std::vector<bool>         * m_l1tResults;
  const L1TUtmTriggerMenu         * m_l1tMenu;
  unsigned long long                m_l1tCacheID;
  bool                              m_l1tUpdated;

  // hlt values and status
  const edm::TriggerResults       * m_hltResults;
  const edm::TriggerNames         * m_hltMenu;
  edm::ParameterSetID               m_hltCacheID;
  bool                              m_hltUpdated;

  // event values
  edm::EventNumber_t                m_eventNumber;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionData_h
