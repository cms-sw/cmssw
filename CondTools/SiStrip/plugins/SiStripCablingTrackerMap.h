#ifndef SiStripCablingTrackerMap_h
#define SiStripCablingTrackerMap_h

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

class SiStripCablingTrackerMap : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  SiStripCablingTrackerMap(const edm::ParameterSet& conf);
  ~SiStripCablingTrackerMap() override = default;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endJob() override;
  void endRun(const edm::Run& run, const edm::EventSetup& es) override{};
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;

  std::unique_ptr<TrackerMap> tkMap_detCab;  //0 for onTrack, 1 for offTrack, 2 for All
};

#endif
