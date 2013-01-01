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
class L1GtTriggerMenu;
class L1GtTriggerMask;

namespace triggerExpression {

class Data {
public:
  // default c'tor
  Data() :
    // configuration
    m_hltResultsTag(""),
    m_l1tResultsTag(""),
    m_daqPartitions(0x01),
    m_l1tIgnoreMask(false),
    m_l1techIgnorePrescales(false),
    m_throw(true),
    // l1 values and status
    m_l1tResults(0),
    m_l1tMenu(0),
    m_l1tAlgoMask(0),
    m_l1tTechMask(0),
    m_l1tCacheID(),
    m_l1tUpdated(false),
    // hlt values and status
    m_hltResults(0),
    m_hltMenu(0),
    m_hltCacheID(),
    m_hltUpdated(false),
    // event values
    m_eventNumber()
  { }

  // explicit c'tor from a ParameterSet
  explicit Data(const edm::ParameterSet & config) :
    // configuration
    m_hltResultsTag(config.getParameter<edm::InputTag>("hltResults")),
    m_l1tResultsTag(config.getParameter<edm::InputTag>("l1tResults")),
    m_daqPartitions(config.getParameter<unsigned int>("daqPartitions")),
    m_l1tIgnoreMask(config.getParameter<bool>("l1tIgnoreMask")),
    m_l1techIgnorePrescales(config.getParameter<bool>("l1techIgnorePrescales")),
    m_throw(config.getParameter<bool>("throw")),
    // l1 values and status
    m_l1tResults(0),
    m_l1tMenu(0),
    m_l1tAlgoMask(0),
    m_l1tTechMask(0),
    m_l1tCacheID(),
    m_l1tUpdated(false),
    // hlt values and status
    m_hltResults(0),
    m_hltMenu(0),
    m_hltCacheID(),
    m_hltUpdated(false),
    // event values
    m_eventNumber()
  { }

  // explicit c'tor from single arguments
  Data(
    edm::InputTag const & hltResultsTag,
    edm::InputTag const & l1tResultsTag,
    unsigned int          daqPartitions,
    bool                  l1tIgnoreMask,
    bool                  l1techIgnorePrescales,
    bool                  doThrow
  ) :
    // configuration
    m_hltResultsTag(hltResultsTag),
    m_l1tResultsTag(l1tResultsTag),
    m_daqPartitions(daqPartitions),
    m_l1tIgnoreMask(l1tIgnoreMask),
    m_l1techIgnorePrescales(l1techIgnorePrescales),
    m_throw(doThrow),
    // l1 values and status
    m_l1tResults(0),
    m_l1tMenu(0),
    m_l1tAlgoMask(0),
    m_l1tTechMask(0),
    m_l1tCacheID(),
    m_l1tUpdated(false),
    // hlt values and status
    m_hltResults(0),
    m_hltMenu(0),
    m_hltCacheID(),
    m_hltUpdated(false),
    // event values
    m_eventNumber()
  { }

  // set the new event
  bool setEvent(const edm::Event & event, const edm::EventSetup & setup);

  // re-configuration accessors 

  void setHltResultsTag(edm::InputTag const & tag) {
    m_hltResultsTag = tag;
  }

  void setL1tResultsTag(edm::InputTag const & tag) {
    m_l1tResultsTag = tag;
  }

  void setDaqPartitions(unsigned int daqPartitions) {
    m_daqPartitions = daqPartitions;
  }

  void setL1tIgnoreMask(bool l1tIgnoreMask) {
    m_l1tIgnoreMask = l1tIgnoreMask;
  }

  void setL1techIgnorePrescales(bool l1techIgnorePrescales) {
    m_l1techIgnorePrescales = l1techIgnorePrescales;
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

  const L1GlobalTriggerReadoutRecord & l1tResults() const {
    return * m_l1tResults;
  }

  const L1GtTriggerMenu & l1tMenu() const {
    return * m_l1tMenu;
  }

  const L1GtTriggerMask & l1tAlgoMask() const {
    return * m_l1tAlgoMask;
  }

  const L1GtTriggerMask & l1tTechMask() const {
    return * m_l1tTechMask;
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

  bool ignoreL1Mask() const {
    return m_l1tIgnoreMask;
  }

  bool ignoreL1TechPrescales() const {
    return m_l1techIgnorePrescales;
  }

  unsigned int daqPartitions() const {
    return m_daqPartitions;
  }

private:
  // configuration
  edm::InputTag m_hltResultsTag;
  edm::InputTag m_l1tResultsTag;
  unsigned int  m_daqPartitions;
  bool          m_l1tIgnoreMask;
  bool          m_l1techIgnorePrescales;
  bool          m_throw;

  // l1 values and status
  const L1GlobalTriggerReadoutRecord  * m_l1tResults;
  const L1GtTriggerMenu               * m_l1tMenu;
  const L1GtTriggerMask               * m_l1tAlgoMask;
  const L1GtTriggerMask               * m_l1tTechMask;
  unsigned long long                    m_l1tCacheID;
  bool                                  m_l1tUpdated;

  // hlt values and status
  const edm::TriggerResults           * m_hltResults;
  const edm::TriggerNames             * m_hltMenu;
  edm::ParameterSetID                   m_hltCacheID;
  bool                                  m_hltUpdated;

  // event values
  edm::EventNumber_t                    m_eventNumber;
};

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionData_h
