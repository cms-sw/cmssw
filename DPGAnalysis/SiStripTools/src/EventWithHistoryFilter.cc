#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"
#include "DPGAnalysis/SiStripTools/interface/APVLatencyRcd.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistoryFilter.h"


EventWithHistoryFilter::EventWithHistoryFilter():
  _historyProduct(),
  _partition("None"), 
  _dbxrange(), _dbxrangelat(),
  _bxrange(), _bxrangelat(),
  _bxcyclerange(), _bxcyclerangelat(),
  _dbxcyclerange(), _dbxcyclerangelat(),
  _dbxtrpltrange(),
  _noAPVPhase(true) 
{
  printConfig();
}

EventWithHistoryFilter::EventWithHistoryFilter(const edm::ParameterSet& iConfig):
  _historyProduct(iConfig.getUntrackedParameter<edm::InputTag>("historyProduct",edm::InputTag("consecutiveHEs"))),
  _partition(iConfig.getUntrackedParameter<std::string>("partitionName","None")),
  _dbxrange(iConfig.getUntrackedParameter<std::vector<int> >("dbxRange",std::vector<int>())),
  _dbxrangelat(iConfig.getUntrackedParameter<std::vector<int> >("dbxRangeLtcyAware",std::vector<int>())),
  _bxrange(iConfig.getUntrackedParameter<std::vector<int> >("absBXRange",std::vector<int>())),
  _bxrangelat(iConfig.getUntrackedParameter<std::vector<int> >("absBXRangeLtcyAware",std::vector<int>())),
  _bxcyclerange(iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRange",std::vector<int>())),
  _bxcyclerangelat(iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRangeLtcyAware",std::vector<int>())),
  _dbxcyclerange(iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRange",std::vector<int>())),
  _dbxcyclerangelat(iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRangeLtcyAware",std::vector<int>())),
  _dbxtrpltrange(iConfig.getUntrackedParameter<std::vector<int> >("dbxTripletRange",std::vector<int>())) 
{
  _noAPVPhase = isAPVPhaseNotNeeded();
  printConfig();
}

void EventWithHistoryFilter::set(const edm::ParameterSet& iConfig) {


  _historyProduct = iConfig.getUntrackedParameter<edm::InputTag>("historyProduct",edm::InputTag("consecutiveHEs"));
  _partition = iConfig.getUntrackedParameter<std::string>("partitionName","None");
  _dbxrange = iConfig.getUntrackedParameter<std::vector<int> >("dbxRange",std::vector<int>());
  _dbxrangelat = iConfig.getUntrackedParameter<std::vector<int> >("dbxRangeLtcyAware",std::vector<int>());
  _bxrange = iConfig.getUntrackedParameter<std::vector<int> >("absBXRange",std::vector<int>());
  _bxrangelat = iConfig.getUntrackedParameter<std::vector<int> >("absBXRangeLtcyAware",std::vector<int>());
  _bxcyclerange = iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRange",std::vector<int>());
  _bxcyclerangelat = iConfig.getUntrackedParameter<std::vector<int> >("absBXInCycleRangeLtcyAware",std::vector<int>());
  _dbxcyclerange = iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRange",std::vector<int>());
  _dbxcyclerangelat = iConfig.getUntrackedParameter<std::vector<int> >("dbxInCycleRangeLtcyAware",std::vector<int>());
  _dbxtrpltrange = iConfig.getUntrackedParameter<std::vector<int> >("dbxTripletRange",std::vector<int>());

  _noAPVPhase = isAPVPhaseNotNeeded();
  printConfig();
 
}

const bool EventWithHistoryFilter::selected(const EventWithHistory& he, const edm::EventSetup& iSetup) const {

  return is_selected(he,iSetup,-1);

}

const bool EventWithHistoryFilter::selected(const EventWithHistory& he, const edm::Run& iRun, const edm::EventSetup& iSetup) const {

  const int apvphase = getAPVPhase(iRun);
  return is_selected(he,iSetup,apvphase);

}

const bool EventWithHistoryFilter::selected(const edm::Event& event, const edm::EventSetup& iSetup) const {

  const int apvphase = getAPVPhase(event.getRun());

  edm::Handle<EventWithHistory> hEvent;
  event.getByLabel(_historyProduct,hEvent);

  return is_selected(*hEvent,iSetup,apvphase);

}


const bool EventWithHistoryFilter::is_selected(const EventWithHistory& he, const edm::EventSetup& iSetup, const int apvphase) const {

  const int latency = getAPVLatency(iSetup);

  bool selected = true;

  selected = selected && (isCutInactive(_dbxrange) || isInRange(he.deltaBX(),_dbxrange,he.depth()!=0));

  selected = selected && (isCutInactive(_dbxrangelat) || 
			  isInRange(he.deltaBX()-latency,_dbxrangelat,he.depth()!=0 && latency>=0));

  selected = selected && (isCutInactive(_bxrange) || isInRange(he.absoluteBX()%70,_bxrange,1));
  
  selected = selected && (isCutInactive(_bxrangelat) || 
			  isInRange((he.absoluteBX()-latency)%70,_bxrangelat,latency>=0));

  selected = selected && (isCutInactive(_bxcyclerange) || 
			  isInRange(he.absoluteBXinCycle(apvphase)%70,_bxcyclerange,apvphase>0));

  selected = selected && (isCutInactive(_bxcyclerangelat) || 
			  isInRange((he.absoluteBXinCycle(apvphase)-latency)%70,_bxcyclerangelat,
				    apvphase>=0 && latency>=0));

  selected = selected && (isCutInactive(_dbxcyclerange) ||
			  isInRange(he.deltaBXinCycle(apvphase),_dbxcyclerange,he.depth()!=0 && apvphase>=0));

  selected = selected && (isCutInactive(_dbxcyclerangelat) ||
			  isInRange(he.deltaBXinCycle(apvphase)-latency,_dbxcyclerangelat,
				    he.depth()!=0 && apvphase>=0 && latency>=0));

  selected = selected && (isCutInactive(_dbxtrpltrange) ||
			  isInRange(he.deltaBX(1,2),_dbxtrpltrange,he.depth()>1));

  return selected;

}

const int EventWithHistoryFilter::getAPVLatency(const edm::EventSetup& iSetup) const {

  if(isAPVLatencyNotNeeded()) return -1;

  edm::ESHandle<APVLatency> apvlat;
  iSetup.get<APVLatencyRcd>().get(apvlat);
  const int latency = apvlat->get();

  // thrown an exception if latency value is invalid

  if(latency < 0) throw cms::Exception("InvalidAPVLatency") << " invalid APVLatency found ";

  return latency;

}

const int EventWithHistoryFilter::getAPVPhase(const edm::Run& iRun) const {

  if(_noAPVPhase) return -1;

  edm::Handle<APVCyclePhaseCollection> apvPhases;
  iRun.getByLabel("APVPhases",apvPhases);

  //  if(!apvPhases.failedToGet() && apvPhases.isValid()) {

  for(std::map<std::string,int>::const_iterator it=apvPhases->get().begin();it!=apvPhases->get().end();it++) {
    if(strstr(it->first.c_str(),_partition.c_str())) return it->second;
  }

  if(!_noAPVPhase) throw cms::Exception("InvalidAPVPhase") 
    << " invalid APV phase found: check if a proper partition has been chosen";

  return -1;
}

const bool EventWithHistoryFilter::isAPVLatencyNotNeeded() const {

  return 
    isCutInactive(_bxrangelat) &&
    isCutInactive(_dbxrangelat) &&
    isCutInactive(_bxcyclerangelat) &&
    isCutInactive(_dbxcyclerangelat);
  
}

const bool EventWithHistoryFilter::isAPVPhaseNotNeeded() const {

  return 
    isCutInactive(_bxcyclerange) &&
    isCutInactive(_dbxcyclerange) &&
    isCutInactive(_bxcyclerangelat) &&
    isCutInactive(_dbxcyclerangelat);
  
}

const bool EventWithHistoryFilter::isCutInactive(const std::vector<int>& range) const {

  return ((range.size()==0 || 
	   (range.size()==1 && range[0]<0) ||   
	   (range.size()==2 && range[0]<0 && range[1]<0)));

}

const bool EventWithHistoryFilter::isInRange(const long long bx, const std::vector<int>& range, const bool extra) const {

  bool cut1 = range.size()<1 || range[0]<0 || (extra && bx >= range[0]);
  bool cut2 = range.size()<2 || range[1]<0 || (extra && bx <= range[1]);
  
  if(range.size()>=2 && range[0]>=0 && range[1]>=0 && (range[0] > range[1])) {
    return cut1 || cut2;
  }
  else {
    return cut1 && cut2;
  }
  
}

void EventWithHistoryFilter::printConfig() const {

  edm::LogInfo("EventWithHistoryFilterConfig");

  edm::LogVerbatim("CutValues") << "-----------------------";
  edm::LogVerbatim("CutValues") << "List of active cuts:";
  if(!isCutInactive(_bxrange)) {
    edm::LogVerbatim("CutValues") << "......................";
    if(_bxrange.size()>=1) edm::LogVerbatim("absoluteBX") << "absoluteBX lower limit " << _bxrange[0];
    if(_bxrange.size()>=2) edm::LogVerbatim("absoluteBX") << "absoluteBX upper limit " << _bxrange[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_bxrangelat)) {
    edm::LogVerbatim("CutValues") << "......................";
    if(_bxrangelat.size()>=1) edm::LogVerbatim("absoluteBXLtcyAware") << "absoluteBXLtcyAware lower limit " 
								      << _bxrangelat[0];
    if(_bxrangelat.size()>=2) edm::LogVerbatim("absoluteBXLtcyAware") << "absoluteBXLtcyAware upper limit " 
								      << _bxrangelat[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_bxcyclerange)) {
    edm::LogVerbatim("CutValues") << "......................";
    edm::LogVerbatim("CutValues") <<"absoluteBXinCycle partition: " << _partition;
    if(_bxcyclerange.size()>=1) edm::LogVerbatim("absoluteBXinCycle") << "absoluteBXinCycle lower limit " 
								      << _bxcyclerange[0];
    if(_bxcyclerange.size()>=2) edm::LogVerbatim("absoluteBXinCycle") << "absoluteBXinCycle upper limit " 
								      << _bxcyclerange[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_bxcyclerangelat)) {
    edm::LogVerbatim("CutValues") << "......................";
    edm::LogVerbatim("CutValues") <<"absoluteBXinCycleLtcyAware partition: " << _partition;
    if(_bxcyclerangelat.size()>=1) edm::LogVerbatim("absoluteBXinCycleLtcyAware") << "absoluteBXinCycleLtcyAware lower limit " 
										  << _bxcyclerangelat[0];
    if(_bxcyclerangelat.size()>=2) edm::LogVerbatim("absoluteBXinCycleLtcyAware") << "absoluteBXinCycleLtcyAware upper limit " 
										  << _bxcyclerangelat[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_dbxrange)) {
    edm::LogVerbatim("CutValues") << "......................";
    if(_dbxrange.size()>=1) edm::LogVerbatim("deltaBX") << "deltaBX lower limit " << _dbxrange[0];
    if(_dbxrange.size()>=2) edm::LogVerbatim("deltaBX") << "deltaBX upper limit " << _dbxrange[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_dbxrangelat)) {
    edm::LogVerbatim("CutValues") << "......................";
    if(_dbxrangelat.size()>=1) edm::LogVerbatim("deltaBXLtcyAware") << "deltaBXLtcyAware lower limit " 
								    << _dbxrangelat[0];
    if(_dbxrangelat.size()>=2) edm::LogVerbatim("deltaBXLtcyAware") << "deltaBXLtcyAware upper limit " 
								    << _dbxrangelat[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_dbxcyclerange)) {
    edm::LogVerbatim("CutValues") << "......................";
    edm::LogVerbatim("CutValues") <<"deltaBXinCycle partition: " << _partition;
    if(_dbxcyclerange.size()>=1) edm::LogVerbatim("deltaBXinCycle") << "deltaBXinCycle lower limit " 
								    << _dbxcyclerange[0];
    if(_dbxcyclerange.size()>=2) edm::LogVerbatim("deltaBXinCycle") << "deltaBXinCycle upper limit " 
								    << _dbxcyclerange[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_dbxcyclerangelat)) {
    edm::LogVerbatim("CutValues") << "......................";
    edm::LogVerbatim("CutValues") <<"deltaBXinCycleLtcyAware partition: " << _partition;
    if(_dbxcyclerangelat.size()>=1) edm::LogVerbatim("deltaBXinCycleLtcyAware") << "deltaBXinCycleLtcyAware lower limit " 
										  << _dbxcyclerangelat[0];
    if(_dbxcyclerangelat.size()>=2) edm::LogVerbatim("deltaBXinCycleLtcyAware") << "deltaBXinCycleLtcyAware upper limit " 
										  << _dbxcyclerangelat[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  if(!isCutInactive(_dbxtrpltrange)) {
    edm::LogVerbatim("CutValues") << "......................";
    if(_dbxtrpltrange.size()>=1) edm::LogVerbatim("TripletIsolation") << "TripletIsolation lower limit " 
								      << _dbxtrpltrange[0];
    if(_dbxtrpltrange.size()>=2) edm::LogVerbatim("TripletIsolation") << "TripletIsolation upper limit " 
								      << _dbxtrpltrange[1];
    edm::LogVerbatim("CutValues") << "......................";
  }
  edm::LogVerbatim("CutValues") << "-----------------------";

}  
