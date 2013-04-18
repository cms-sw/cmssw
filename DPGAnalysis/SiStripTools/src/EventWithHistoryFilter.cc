#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
//#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"
//#include "DPGAnalysis/SiStripTools/interface/APVLatencyRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Utilities/interface/Exception.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistoryFilter.h"


EventWithHistoryFilter::EventWithHistoryFilter():
  _historyProduct(),
  _partition(), 
  _APVPhaseLabel(),
  _apvmodes(),
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
  _partition(iConfig.getUntrackedParameter<std::string>("partitionName","Any")),
  _APVPhaseLabel(iConfig.getUntrackedParameter<std::string>("APVPhaseLabel","APVPhases")),
  _apvmodes(iConfig.getUntrackedParameter<std::vector<int> >("apvModes",std::vector<int>())),
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
  _partition = iConfig.getUntrackedParameter<std::string>("partitionName","Any");
  _APVPhaseLabel = iConfig.getUntrackedParameter<std::string>("APVPhaseLabel","APVPhases");
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

  const std::vector<int> dummy;
  return is_selected(he,iSetup,dummy);

}

const bool EventWithHistoryFilter::selected(const EventWithHistory& he, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {

  const std::vector<int> apvphases = getAPVPhase(iEvent);
  return is_selected(he,iSetup,apvphases);

}

const bool EventWithHistoryFilter::selected(const edm::Event& event, const edm::EventSetup& iSetup) const {

  const std::vector<int> apvphases = getAPVPhase(event);

  edm::Handle<EventWithHistory> hEvent;
  event.getByLabel(_historyProduct,hEvent);

  return is_selected(*hEvent,iSetup,apvphases);

}


const bool EventWithHistoryFilter::is_selected(const EventWithHistory& he, const edm::EventSetup& iSetup, const std::vector<int> apvphases) const {


  const int latency = getAPVLatency(iSetup);


  bool selected = true;

  if(!isAPVModeNotNeeded()) {
    const int apvmode = getAPVMode(iSetup);
    bool modeok = false;
    for(std::vector<int>::const_iterator wantedmode =_apvmodes.begin();wantedmode!=_apvmodes.end();++wantedmode) {
      modeok = modeok || (apvmode == *wantedmode);
    }
    if(!modeok) return false;
  }

  selected = selected && (isCutInactive(_dbxrange) || isInRange(he.deltaBX(),_dbxrange,he.depth()!=0));

  selected = selected && (isCutInactive(_dbxrangelat) || 
			  isInRange(he.deltaBX()-latency,_dbxrangelat,he.depth()!=0 && latency>=0));

  selected = selected && (isCutInactive(_bxrange) || isInRange(he.absoluteBX()%70,_bxrange,1));
  
  selected = selected && (isCutInactive(_bxrangelat) || 
			  isInRange((he.absoluteBX()-latency)%70,_bxrangelat,latency>=0));

  // loop on all the phases and require that the cut is fulfilled for at least one of them

  
  bool phaseselected;

  phaseselected = isCutInactive(_bxcyclerange);
  for(std::vector<int>::const_iterator phase=apvphases.begin();phase!=apvphases.end();++phase) {
    phaseselected = phaseselected || isInRange(he.absoluteBXinCycle(*phase)%70,_bxcyclerange,*phase>0);
  }
  selected = selected && phaseselected;
    
  phaseselected = isCutInactive(_bxcyclerangelat);
  for(std::vector<int>::const_iterator phase=apvphases.begin();phase!=apvphases.end();++phase) {
    phaseselected = phaseselected || isInRange((he.absoluteBXinCycle(*phase)-latency)%70,_bxcyclerangelat,
					       *phase>=0 && latency>=0);
  }
  selected = selected && phaseselected;
    
  phaseselected = isCutInactive(_dbxcyclerange);
  for(std::vector<int>::const_iterator phase=apvphases.begin();phase!=apvphases.end();++phase) {
    phaseselected = phaseselected || isInRange(he.deltaBXinCycle(*phase),_dbxcyclerange,he.depth()!=0 && *phase>=0);
  }
  selected = selected && phaseselected;
    
  phaseselected = isCutInactive(_dbxcyclerangelat);
  for(std::vector<int>::const_iterator phase=apvphases.begin();phase!=apvphases.end();++phase) {
    phaseselected = phaseselected || isInRange(he.deltaBXinCycle(*phase)-latency,_dbxcyclerangelat,
					       he.depth()!=0 && *phase>=0 && latency>=0);
  }
  selected = selected && phaseselected;
    
  // end of phase-dependent cuts

  selected = selected && (isCutInactive(_dbxtrpltrange) ||
			  isInRange(he.deltaBX(1,2),_dbxtrpltrange,he.depth()>1));

  return selected;

}

const int EventWithHistoryFilter::getAPVLatency(const edm::EventSetup& iSetup) const {

  if(isAPVLatencyNotNeeded()) return -1;

  edm::ESHandle<SiStripLatency> apvlat;
  iSetup.get<SiStripLatencyRcd>().get(apvlat);
  const int latency = apvlat->singleLatency()!=255 ? apvlat->singleLatency(): -1;

  // thrown an exception if latency value is invalid
  /*
  if(latency < 0  && !isAPVLatencyNotNeeded()) 
    throw cms::Exception("InvalidAPVLatency") << " invalid APVLatency found ";
  */

  return latency;

}

const int EventWithHistoryFilter::getAPVMode(const edm::EventSetup& iSetup) const {

  if(isAPVModeNotNeeded()) return -1;

  edm::ESHandle<SiStripLatency> apvlat;
  iSetup.get<SiStripLatencyRcd>().get(apvlat);
  int mode = -1;
  if(apvlat->singleReadOutMode()==1) mode = 47;
  if(apvlat->singleReadOutMode()==0) mode = 37;

  // thrown an exception if mode value is invalid
  /*
  if(mode < 0 && !isAPVModeNotNeeded()) 
    throw cms::Exception("InvalidAPVMode") << " invalid APVMode found ";
  */

  return mode;

}

const std::vector<int> EventWithHistoryFilter::getAPVPhase(const edm::Event& iEvent) const {

  if(_noAPVPhase) {
    const std::vector<int> dummy;
    return dummy;
  }

  edm::Handle<APVCyclePhaseCollection> apvPhases;
  iEvent.getByLabel(_APVPhaseLabel,apvPhases);

  const std::vector<int> phases = apvPhases->getPhases(_partition.c_str());

  /*
  if(!_noAPVPhase) {
    if(phases.size()==0) throw cms::Exception("NoPartitionAPVPhase") 
      << " No APV phase for partition " << _partition.c_str() << " : check if a proper partition has been chosen ";
  }
  */

  return phases;
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

const bool EventWithHistoryFilter::isAPVModeNotNeeded() const {

  return (_apvmodes.size()==0) ;
  
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

  std::string msgcategory = "EventWithHistoryFilterConfiguration";

  if(!(
       isCutInactive(_bxrange) &&
       isCutInactive(_bxrangelat) &&
       isCutInactive(_bxcyclerange) &&
       isCutInactive(_bxcyclerangelat) &&
       isCutInactive(_dbxrange) &&
       isCutInactive(_dbxrangelat) &&
       isCutInactive(_dbxcyclerange) &&
       isCutInactive(_dbxcyclerangelat) &&
       isCutInactive(_dbxtrpltrange)
       )) {

    edm::LogInfo(msgcategory.c_str()) << "historyProduct: " 
							<< _historyProduct.label() << " " 
							<< _historyProduct.instance() << " " 
							<< _historyProduct.process() << " " 
							<< " APVCyclePhase: " 
							<< _APVPhaseLabel;
    
    edm::LogVerbatim(msgcategory.c_str()) << "-----------------------";
    edm::LogVerbatim(msgcategory.c_str()) << "List of active cuts:";
    if(!isCutInactive(_bxrange)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      if(_bxrange.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBX lower limit " << _bxrange[0];
      if(_bxrange.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBX upper limit " << _bxrange[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_bxrangelat)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      if(_bxrangelat.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBXLtcyAware lower limit " 
									<< _bxrangelat[0];
      if(_bxrangelat.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBXLtcyAware upper limit " 
									<< _bxrangelat[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_bxcyclerange)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      edm::LogVerbatim(msgcategory.c_str()) <<"absoluteBXinCycle partition: " << _partition;
      if(_bxcyclerange.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBXinCycle lower limit " 
									<< _bxcyclerange[0];
      if(_bxcyclerange.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBXinCycle upper limit " 
									<< _bxcyclerange[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_bxcyclerangelat)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      edm::LogVerbatim(msgcategory.c_str()) <<"absoluteBXinCycleLtcyAware partition: " << _partition;
      if(_bxcyclerangelat.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBXinCycleLtcyAware lower limit " 
										    << _bxcyclerangelat[0];
      if(_bxcyclerangelat.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "absoluteBXinCycleLtcyAware upper limit " 
										    << _bxcyclerangelat[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_dbxrange)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      if(_dbxrange.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "deltaBX lower limit " << _dbxrange[0];
      if(_dbxrange.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "deltaBX upper limit " << _dbxrange[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_dbxrangelat)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      if(_dbxrangelat.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "deltaBXLtcyAware lower limit " 
								      << _dbxrangelat[0];
      if(_dbxrangelat.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "deltaBXLtcyAware upper limit " 
								      << _dbxrangelat[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_dbxcyclerange)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      edm::LogVerbatim(msgcategory.c_str()) <<"deltaBXinCycle partition: " << _partition;
      if(_dbxcyclerange.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "deltaBXinCycle lower limit " 
								      << _dbxcyclerange[0];
      if(_dbxcyclerange.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "deltaBXinCycle upper limit " 
								      << _dbxcyclerange[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_dbxcyclerangelat)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      edm::LogVerbatim(msgcategory.c_str()) <<"deltaBXinCycleLtcyAware partition: " << _partition;
      if(_dbxcyclerangelat.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "deltaBXinCycleLtcyAware lower limit " 
										  << _dbxcyclerangelat[0];
      if(_dbxcyclerangelat.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "deltaBXinCycleLtcyAware upper limit " 
										  << _dbxcyclerangelat[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    if(!isCutInactive(_dbxtrpltrange)) {
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
      if(_dbxtrpltrange.size()>=1) edm::LogVerbatim(msgcategory.c_str()) << "TripletIsolation lower limit " 
									<< _dbxtrpltrange[0];
      if(_dbxtrpltrange.size()>=2) edm::LogVerbatim(msgcategory.c_str()) << "TripletIsolation upper limit " 
									<< _dbxtrpltrange[1];
      edm::LogVerbatim(msgcategory.c_str()) << "......................";
    }
    edm::LogVerbatim(msgcategory.c_str()) << "-----------------------";
  }
}  
