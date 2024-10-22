#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"
//#include "DPGAnalysis/SiStripTools/interface/APVLatencyRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Utilities/interface/Exception.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistoryFilter.h"

EventWithHistoryFilter::EventWithHistoryFilter()
    : m_historyToken(),
      m_partition(),
      m_APVPhaseToken(),
      m_apvmodes(),
      m_dbxrange(),
      m_dbxrangelat(),
      m_bxrange(),
      m_bxrangelat(),
      m_bxcyclerange(),
      m_bxcyclerangelat(),
      m_dbxcyclerange(),
      m_dbxcyclerangelat(),
      m_dbxtrpltrange(),
      m_dbxgenericrange(),
      m_dbxgenericfirst(0),
      m_dbxgenericlast(1),
      m_noAPVPhase(true) {
  printConfig(edm::InputTag(), edm::InputTag());
}

EventWithHistoryFilter::EventWithHistoryFilter(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : m_historyToken(iC.consumes<EventWithHistory>(
          iConfig.getUntrackedParameter<edm::InputTag>("historyProduct", edm::InputTag("consecutiveHEs")))),
      m_partition(iConfig.getUntrackedParameter<std::string>("partitionName", "Any")),
      m_APVPhaseToken(iC.consumes<APVCyclePhaseCollection>(
          edm::InputTag(iConfig.getUntrackedParameter<std::string>("APVPhaseLabel", "APVPhases")))),
      m_apvLatencyToken(iC.esConsumes<SiStripLatency, SiStripLatencyRcd>()),
      m_apvmodes(iConfig.getUntrackedParameter<std::vector<int> >("apvModes", std::vector<int>())),
      m_dbxrange(iConfig.getUntrackedParameter<std::vector<int> >("dbxRange", std::vector<int>())),
      m_dbxrangelat(iConfig.getUntrackedParameter<std::vector<int> >("dbxRangeLtcyAware", std::vector<int>())),
      m_bxrange(iConfig.getUntrackedParameter<std::vector<int> >("absBXRange", std::vector<int>())),
      m_bxrangelat(iConfig.getUntrackedParameter<std::vector<int> >("absBXRangeLtcyAware", std::vector<int>())),
      m_bxcyclerange(iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRange", std::vector<int>())),
      m_bxcyclerangelat(
          iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRangeLtcyAware", std::vector<int>())),
      m_dbxcyclerange(iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRange", std::vector<int>())),
      m_dbxcyclerangelat(
          iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRangeLtcyAware", std::vector<int>())),
      m_dbxtrpltrange(iConfig.getUntrackedParameter<std::vector<int> >("dbxTripletRange", std::vector<int>())),
      m_dbxgenericrange(iConfig.getUntrackedParameter<std::vector<int> >("dbxGenericRange", std::vector<int>())),
      m_dbxgenericfirst(iConfig.getUntrackedParameter<unsigned int>("dbxGenericFirst", 0)),
      m_dbxgenericlast(iConfig.getUntrackedParameter<unsigned int>("dbxGenericLast", 1))

{
  m_noAPVPhase = isAPVPhaseNotNeeded();
  printConfig(iConfig.getUntrackedParameter<edm::InputTag>("historyProduct", edm::InputTag("consecutiveHEs")),
              edm::InputTag(iConfig.getUntrackedParameter<std::string>("APVPhaseLabel", "APVPhases")));
}

void EventWithHistoryFilter::set(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC) {
  m_historyToken = iC.consumes<EventWithHistory>(
      iConfig.getUntrackedParameter<edm::InputTag>("historyProduct", edm::InputTag("consecutiveHEs")));
  m_partition = iConfig.getUntrackedParameter<std::string>("partitionName", "Any");
  m_APVPhaseToken = iC.consumes<APVCyclePhaseCollection>(
      edm::InputTag(iConfig.getUntrackedParameter<std::string>("APVPhaseLabel", "APVPhases")));
  m_apvLatencyToken = iC.esConsumes<SiStripLatency, SiStripLatencyRcd>();
  m_dbxrange = iConfig.getUntrackedParameter<std::vector<int> >("dbxRange", std::vector<int>());
  m_dbxrangelat = iConfig.getUntrackedParameter<std::vector<int> >("dbxRangeLtcyAware", std::vector<int>());
  m_bxrange = iConfig.getUntrackedParameter<std::vector<int> >("absBXRange", std::vector<int>());
  m_bxrangelat = iConfig.getUntrackedParameter<std::vector<int> >("absBXRangeLtcyAware", std::vector<int>());
  m_bxcyclerange = iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRange", std::vector<int>());
  m_bxcyclerangelat =
      iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRangeLtcyAware", std::vector<int>());
  m_dbxcyclerange = iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRange", std::vector<int>());
  m_dbxcyclerangelat = iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRangeLtcyAware", std::vector<int>());
  m_dbxtrpltrange = iConfig.getUntrackedParameter<std::vector<int> >("dbxTripletRange", std::vector<int>());
  m_dbxgenericrange = iConfig.getUntrackedParameter<std::vector<int> >("dbxGenericRange", std::vector<int>());
  m_dbxgenericfirst = iConfig.getUntrackedParameter<int>("dbxGenericFirst", 0);
  m_dbxgenericlast = iConfig.getUntrackedParameter<int>("dbxGenericLast", 1);

  m_noAPVPhase = isAPVPhaseNotNeeded();
  printConfig(iConfig.getUntrackedParameter<edm::InputTag>("historyProduct", edm::InputTag("consecutiveHEs")),
              edm::InputTag(iConfig.getUntrackedParameter<std::string>("APVPhaseLabel", "APVPhases")));
}

const bool EventWithHistoryFilter::selected(const EventWithHistory& he, const edm::EventSetup& iSetup) const {
  const std::vector<int> dummy;
  return is_selected(he, iSetup, dummy);
}

const bool EventWithHistoryFilter::selected(const EventWithHistory& he,
                                            const edm::Event& iEvent,
                                            const edm::EventSetup& iSetup) const {
  const std::vector<int> apvphases = getAPVPhase(iEvent);
  return is_selected(he, iSetup, apvphases);
}

const bool EventWithHistoryFilter::selected(const edm::Event& event, const edm::EventSetup& iSetup) const {
  const std::vector<int> apvphases = getAPVPhase(event);

  edm::Handle<EventWithHistory> hEvent;
  event.getByToken(m_historyToken, hEvent);

  return is_selected(*hEvent, iSetup, apvphases);
}

const bool EventWithHistoryFilter::is_selected(const EventWithHistory& he,
                                               const edm::EventSetup& iSetup,
                                               const std::vector<int>& _apvphases) const {
  const std::vector<int>& apvphases = _apvphases;
  const int latency = getAPVLatency(iSetup);

  bool selected = true;

  if (!isAPVModeNotNeeded()) {
    const int apvmode = getAPVMode(iSetup);
    bool modeok = false;
    for (std::vector<int>::const_iterator wantedmode = m_apvmodes.begin(); wantedmode != m_apvmodes.end();
         ++wantedmode) {
      modeok = modeok || (apvmode == *wantedmode);
    }
    if (!modeok)
      return false;
  }

  selected = selected && (isCutInactive(m_dbxrange) || isInRange(he.deltaBX(), m_dbxrange, he.depth() != 0));

  selected = selected && (isCutInactive(m_dbxrangelat) ||
                          isInRange(he.deltaBX() - latency, m_dbxrangelat, he.depth() != 0 && latency >= 0));

  selected = selected && (isCutInactive(m_bxrange) || isInRange(he.absoluteBX() % 70, m_bxrange, true));

  selected = selected &&
             (isCutInactive(m_bxrangelat) || isInRange((he.absoluteBX() - latency) % 70, m_bxrangelat, latency >= 0));

  // loop on all the phases and require that the cut is fulfilled for at least one of them

  bool phaseselected;

  phaseselected = isCutInactive(m_bxcyclerange);
  for (std::vector<int>::const_iterator phase = apvphases.begin(); phase != apvphases.end(); ++phase) {
    phaseselected = phaseselected || isInRange(he.absoluteBXinCycle(*phase) % 70, m_bxcyclerange, *phase >= 0);
  }
  selected = selected && phaseselected;

  phaseselected = isCutInactive(m_bxcyclerangelat);
  for (std::vector<int>::const_iterator phase = apvphases.begin(); phase != apvphases.end(); ++phase) {
    phaseselected =
        phaseselected ||
        isInRange((he.absoluteBXinCycle(*phase) - latency) % 70, m_bxcyclerangelat, *phase >= 0 && latency >= 0);
  }
  selected = selected && phaseselected;

  phaseselected = isCutInactive(m_dbxcyclerange);
  for (std::vector<int>::const_iterator phase = apvphases.begin(); phase != apvphases.end(); ++phase) {
    phaseselected =
        phaseselected || isInRange(he.deltaBXinCycle(*phase), m_dbxcyclerange, he.depth() != 0 && *phase >= 0);
  }
  selected = selected && phaseselected;

  phaseselected = isCutInactive(m_dbxcyclerangelat);
  for (std::vector<int>::const_iterator phase = apvphases.begin(); phase != apvphases.end(); ++phase) {
    phaseselected = phaseselected || isInRange(he.deltaBXinCycle(*phase) - latency,
                                               m_dbxcyclerangelat,
                                               he.depth() != 0 && *phase >= 0 && latency >= 0);
  }
  selected = selected && phaseselected;

  // end of phase-dependent cuts

  selected =
      selected && (isCutInactive(m_dbxtrpltrange) || isInRange(he.deltaBX(1, 2), m_dbxtrpltrange, he.depth() > 1));

  selected =
      selected &&
      (isCutInactive(m_dbxgenericrange) ||
       isInRange(he.deltaBX(m_dbxgenericfirst, m_dbxgenericlast), m_dbxgenericrange, he.depth() >= m_dbxgenericlast));

  return selected;
}

const int EventWithHistoryFilter::getAPVLatency(const edm::EventSetup& iSetup) const {
  if (isAPVLatencyNotNeeded())
    return -1;

  const auto& apvlat = iSetup.getData(m_apvLatencyToken);
  const int latency = apvlat.singleLatency() != 255 ? apvlat.singleLatency() : -1;

  // thrown an exception if latency value is invalid
  /*
  if(latency < 0  && !isAPVLatencyNotNeeded())
    throw cms::Exception("InvalidAPVLatency") << " invalid APVLatency found ";
  */

  return latency;
}

const int EventWithHistoryFilter::getAPVMode(const edm::EventSetup& iSetup) const {
  if (isAPVModeNotNeeded())
    return -1;

  const auto& apvlat = iSetup.getData(m_apvLatencyToken);
  int mode = -1;
  if (apvlat.singleReadOutMode() == 1)
    mode = 47;
  if (apvlat.singleReadOutMode() == 0)
    mode = 37;

  // thrown an exception if mode value is invalid
  /*
  if(mode < 0 && !isAPVModeNotNeeded())
    throw cms::Exception("InvalidAPVMode") << " invalid APVMode found ";
  */

  return mode;
}

const std::vector<int> EventWithHistoryFilter::getAPVPhase(const edm::Event& iEvent) const {
  if (m_noAPVPhase) {
    const std::vector<int> dummy;
    return dummy;
  }

  edm::Handle<APVCyclePhaseCollection> apvPhases;
  iEvent.getByToken(m_APVPhaseToken, apvPhases);

  const std::vector<int> phases = apvPhases->getPhases(m_partition);

  /*
  if(!m_noAPVPhase) {
    if(phases.size()==0) throw cms::Exception("NoPartitionAPVPhase")
      << " No APV phase for partition " << m_partition.c_str() << " : check if a proper partition has been chosen ";
  }
  */

  return phases;
}

const bool EventWithHistoryFilter::isAPVLatencyNotNeeded() const {
  return isCutInactive(m_bxrangelat) && isCutInactive(m_dbxrangelat) && isCutInactive(m_bxcyclerangelat) &&
         isCutInactive(m_dbxcyclerangelat);
}

const bool EventWithHistoryFilter::isAPVPhaseNotNeeded() const {
  return isCutInactive(m_bxcyclerange) && isCutInactive(m_dbxcyclerange) && isCutInactive(m_bxcyclerangelat) &&
         isCutInactive(m_dbxcyclerangelat);
}

const bool EventWithHistoryFilter::isAPVModeNotNeeded() const { return (m_apvmodes.empty()); }

const bool EventWithHistoryFilter::isCutInactive(const std::vector<int>& range) const {
  return (
      (range.empty() || (range.size() == 1 && range[0] < 0) || (range.size() == 2 && range[0] < 0 && range[1] < 0)));
}

const bool EventWithHistoryFilter::isInRange(const long long bx,
                                             const std::vector<int>& range,
                                             const bool extra) const {
  bool cut1 = range.empty() || range[0] < 0 || (extra && bx >= range[0]);
  bool cut2 = range.size() < 2 || range[1] < 0 || (extra && bx <= range[1]);

  if (range.size() >= 2 && range[0] >= 0 && range[1] >= 0 && (range[0] > range[1])) {
    return cut1 || cut2;
  } else {
    return cut1 && cut2;
  }
}

void EventWithHistoryFilter::printConfig(const edm::InputTag& historyTag, const edm::InputTag& apvphaseTag) const {
  std::string msgcategory = "EventWithHistoryFilterConfiguration";

  if (!(isCutInactive(m_bxrange) && isCutInactive(m_bxrangelat) && isCutInactive(m_bxcyclerange) &&
        isCutInactive(m_bxcyclerangelat) && isCutInactive(m_dbxrange) && isCutInactive(m_dbxrangelat) &&
        isCutInactive(m_dbxcyclerange) && isCutInactive(m_dbxcyclerangelat) && isCutInactive(m_dbxtrpltrange) &&
        isCutInactive(m_dbxgenericrange))) {
    edm::LogInfo(msgcategory) << "historyProduct: " << historyTag << " APVCyclePhase: " << apvphaseTag;

    edm::LogVerbatim(msgcategory) << "-----------------------";
    edm::LogVerbatim(msgcategory) << "List of active cuts:";
    if (!isCutInactive(m_bxrange)) {
      edm::LogVerbatim(msgcategory) << "......................";
      if (!m_bxrange.empty())
        edm::LogVerbatim(msgcategory) << "absoluteBX lower limit " << m_bxrange[0];
      if (m_bxrange.size() >= 2)
        edm::LogVerbatim(msgcategory) << "absoluteBX upper limit " << m_bxrange[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_bxrangelat)) {
      edm::LogVerbatim(msgcategory) << "......................";
      if (!m_bxrangelat.empty())
        edm::LogVerbatim(msgcategory) << "absoluteBXLtcyAware lower limit " << m_bxrangelat[0];
      if (m_bxrangelat.size() >= 2)
        edm::LogVerbatim(msgcategory) << "absoluteBXLtcyAware upper limit " << m_bxrangelat[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_bxcyclerange)) {
      edm::LogVerbatim(msgcategory) << "......................";
      edm::LogVerbatim(msgcategory) << "absoluteBXinCycle partition: " << m_partition;
      if (!m_bxcyclerange.empty())
        edm::LogVerbatim(msgcategory) << "absoluteBXinCycle lower limit " << m_bxcyclerange[0];
      if (m_bxcyclerange.size() >= 2)
        edm::LogVerbatim(msgcategory) << "absoluteBXinCycle upper limit " << m_bxcyclerange[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_bxcyclerangelat)) {
      edm::LogVerbatim(msgcategory) << "......................";
      edm::LogVerbatim(msgcategory) << "absoluteBXinCycleLtcyAware partition: " << m_partition;
      if (!m_bxcyclerangelat.empty())
        edm::LogVerbatim(msgcategory) << "absoluteBXinCycleLtcyAware lower limit " << m_bxcyclerangelat[0];
      if (m_bxcyclerangelat.size() >= 2)
        edm::LogVerbatim(msgcategory) << "absoluteBXinCycleLtcyAware upper limit " << m_bxcyclerangelat[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_dbxrange)) {
      edm::LogVerbatim(msgcategory) << "......................";
      if (!m_dbxrange.empty())
        edm::LogVerbatim(msgcategory) << "deltaBX lower limit " << m_dbxrange[0];
      if (m_dbxrange.size() >= 2)
        edm::LogVerbatim(msgcategory) << "deltaBX upper limit " << m_dbxrange[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_dbxrangelat)) {
      edm::LogVerbatim(msgcategory) << "......................";
      if (!m_dbxrangelat.empty())
        edm::LogVerbatim(msgcategory) << "deltaBXLtcyAware lower limit " << m_dbxrangelat[0];
      if (m_dbxrangelat.size() >= 2)
        edm::LogVerbatim(msgcategory) << "deltaBXLtcyAware upper limit " << m_dbxrangelat[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_dbxcyclerange)) {
      edm::LogVerbatim(msgcategory) << "......................";
      edm::LogVerbatim(msgcategory) << "deltaBXinCycle partition: " << m_partition;
      if (!m_dbxcyclerange.empty())
        edm::LogVerbatim(msgcategory) << "deltaBXinCycle lower limit " << m_dbxcyclerange[0];
      if (m_dbxcyclerange.size() >= 2)
        edm::LogVerbatim(msgcategory) << "deltaBXinCycle upper limit " << m_dbxcyclerange[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_dbxcyclerangelat)) {
      edm::LogVerbatim(msgcategory) << "......................";
      edm::LogVerbatim(msgcategory) << "deltaBXinCycleLtcyAware partition: " << m_partition;
      if (!m_dbxcyclerangelat.empty())
        edm::LogVerbatim(msgcategory) << "deltaBXinCycleLtcyAware lower limit " << m_dbxcyclerangelat[0];
      if (m_dbxcyclerangelat.size() >= 2)
        edm::LogVerbatim(msgcategory) << "deltaBXinCycleLtcyAware upper limit " << m_dbxcyclerangelat[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_dbxtrpltrange)) {
      edm::LogVerbatim(msgcategory) << "......................";
      if (!m_dbxtrpltrange.empty())
        edm::LogVerbatim(msgcategory) << "TripletIsolation lower limit " << m_dbxtrpltrange[0];
      if (m_dbxtrpltrange.size() >= 2)
        edm::LogVerbatim(msgcategory) << "TripletIsolation upper limit " << m_dbxtrpltrange[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    if (!isCutInactive(m_dbxgenericrange)) {
      edm::LogVerbatim(msgcategory) << "......................";
      edm::LogVerbatim(msgcategory) << "Generic DBX computed between n-" << m_dbxgenericfirst << " and n-"
                                    << m_dbxgenericlast << " trigger";
      if (!m_dbxgenericrange.empty())
        edm::LogVerbatim(msgcategory) << "Generic DBX cut lower limit " << m_dbxgenericrange[0];
      if (m_dbxgenericrange.size() >= 2)
        edm::LogVerbatim(msgcategory) << "Generic DBX upper limit " << m_dbxgenericrange[1];
      edm::LogVerbatim(msgcategory) << "......................";
    }
    edm::LogVerbatim(msgcategory) << "-----------------------";
  }
}
