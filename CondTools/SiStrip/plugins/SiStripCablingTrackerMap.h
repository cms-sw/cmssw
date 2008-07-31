#ifndef SiStripCablingTrackerMap_h
#define SiStripCablingTrackerMap_h

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

class SiStripCablingTrackerMap : public edm::EDAnalyzer
{
  
 public:
  
  SiStripCablingTrackerMap(const edm::ParameterSet& conf);
  ~SiStripCablingTrackerMap();
  
  void beginRun(const edm::Run& run,  const edm::EventSetup& es );
  
  void endJob();
  
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
 private:
  
  edm::ParameterSet conf_;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  
  TrackerMap* tkMap_detCab;//0 for onTrack, 1 for offTrack, 2 for All  
};

#endif
