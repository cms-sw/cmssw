#include "TkMeasurementDetSet.h"
#include "TkStripMeasurementDet.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


void StMeasurementDetSet::init(std::vector<TkStripMeasurementDet> & stripDets) {
  // assume vector is full and ordered!
  int size = stripDets.size();
  
  empty_.resize(size,true);
  activeThisEvent_.resize(size,true);
  activeThisPeriod_.resize(size,true);
  id_.resize(size);
  subId_.resize(size);
  totalStrips_.resize(size);
  
  bad128Strip_.resize(size*6);
  hasAny128StripBad_.resize(size);
  badStripBlocks_.resize(size);

  if (isRegional()) {
    clusterI_.resize(2*size);
  }  else {
    detSet_.resize(size);
  }
  
  for (int i=0; i!=size; ++i) {
    auto & mdet =  stripDets[i]; 
    mdet.setIndex(i);
    //intialize the detId !
    id_[i] = mdet.specificGeomDet().geographicalId().rawId();
    subId_[i]=SiStripDetId(id_[i]).subdetId()-3;
    //initalize the total number of strips
    totalStrips_[i] =  mdet.specificGeomDet().specificTopology().nstrips();
  }
}


void StMeasurementDetSet::set128StripStatus(int i, bool good, int idx) { 
    int offset =  nbad128*i;
    if (idx == -1) {
      std::fill(bad128Strip_.begin()+offset, bad128Strip_.begin()+offset+6, !good);
      hasAny128StripBad_[i] = !good;
    } else {
      bad128Strip_[offset+idx] = !good;
      if (good == false) {
	hasAny128StripBad_[i] = false;
      } else { // this should not happen, as usually you turn on all fibers
	// and then turn off the bad ones, and not vice-versa,
	// so I don't care if it's not optimized
	hasAny128StripBad_[i] = true;
	for (int j = 0; i < (totalStrips_[j] >> 7); j++) {
	  if (bad128Strip_[j+offset] == false) hasAny128StripBad_[i] = false; break;
	}
      }    
    } 
  }
  

void StMeasurementDetSet::initializeStripStatus(const SiStripQuality *quality, int qualityFlags, int qualityDebugFlags, edm::ParameterSet cutPset) {
  if (qualityFlags & BadStrips) {
    edm::ParameterSet cutPset = pset_.getParameter<edm::ParameterSet>("badStripCuts");
    badStripCuts_[SiStripDetId::TIB-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TIB"));
    badStripCuts_[SiStripDetId::TOB-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TOB"));
    badStripCuts_[SiStripDetId::TID-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TID"));
    badStripCuts_[SiStripDetId::TEC-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TEC"));
  }
  setMaskBad128StripBlocks((qualityFlags & MaskBad128StripBlocks) != 0);
  
  
  if ((quality != 0) && (qualityFlags != 0))  {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0; 
    unsigned int foff = 0, ftot = 0, aoff = 0, atot = 0; 
    for (int i=0; i!=nDet(); i++) {
      uint32_t detid = id(i);
      if (qualityFlags & BadModules) {
	bool isOn = quality->IsModuleUsable(detid);
	setActive(i,isOn);
	tot++; on += (unsigned int) isOn;
	if (qualityDebugFlags & BadModules) {
	  edm::LogInfo("MeasurementTracker")<< "MeasurementTrackerImpl::initializeStripStatus : detid " << detid << " is " << (isOn ?  "on" : "off");
	}
      } else {
	setActive(i,true);
      }
      // first turn all APVs and fibers ON
      set128StripStatus(i,true); 
      if (qualityFlags & BadAPVFibers) {
	short badApvs   = quality->getBadApvs(detid);
	short badFibers = quality->getBadFibers(detid);
	for (int j = 0; j < 6; j++) {
	  atot++;
	  if (badApvs & (1 << j)) {
	    set128StripStatus(i,false, j);
	    aoff++;
	  }
	}
	for (int j = 0; j < 3; j++) {
	  ftot++;
             if (badFibers & (1 << j)) {
	       set128StripStatus(i,false, 2*j);
	       set128StripStatus(i,false, 2*j+1);
	       foff++;
             }
	}
      } 
      auto & badStrips = getBadStripBlocks(i);
       badStrips.clear();
       if (qualityFlags & BadStrips) {
	 SiStripBadStrip::Range range = quality->getRange(detid);
	 for (SiStripBadStrip::ContainerIterator bit = range.first; bit != range.second; ++bit) {
	   badStrips.push_back(quality->decode(*bit));
	 }
       }
    }
    if (qualityDebugFlags & BadModules) {
      edm::LogInfo("MeasurementTracker StripModuleStatus") << 
	" Total modules: " << tot << ", active " << on <<", inactive " << (tot - on);
    }
    if (qualityDebugFlags & BadAPVFibers) {
      edm::LogInfo("MeasurementTracker StripAPVStatus") << 
	" Total APVs: " << atot << ", active " << (atot-aoff) <<", inactive " << (aoff);
        edm::LogInfo("MeasurementTracker StripFiberStatus") << 
	  " Total Fibers: " << ftot << ", active " << (ftot-foff) <<", inactive " << (foff);
    }
  } else {
    for (int i=0; i!=nDet(); i++) {
      setActive(i,true);          // module ON
      set128StripStatus(u,true);  // all APVs and fibers ON
    }
  }
}
