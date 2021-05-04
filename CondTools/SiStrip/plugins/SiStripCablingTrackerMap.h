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

class SiStripCablingTrackerMap : public edm::EDAnalyzer {
public:
  SiStripCablingTrackerMap(const edm::ParameterSet& conf);
  ~SiStripCablingTrackerMap() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;

  void endJob() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ParameterSet conf_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;

  TrackerMap* tkMap_detCab;  //0 for onTrack, 1 for offTrack, 2 for All
};

#endif
