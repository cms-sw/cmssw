#ifndef __TrackerDTC_TTDTCCONVERTER_H__
#define __TrackerDTC_TTDTCCONVERTER_H__

#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"

#include <string>
#include <memory>

// converts TTDTC::Frame into GlobalPoint
class TTDTCConverter {
public:
  TTDTCConverter(const edm::Run& iRun,
                 const edm::EventSetup& iSetup,
                 const std::string& processName = "",
                 const std::string& productLabel = "TrackerDTCProducer");

  ~TTDTCConverter(){}

  // returns bit accurate position of a stub from a given processing region [0-8] (phi slice of outer tracker)
  GlobalPoint pos(const TTDTC::Frame& frame, const int& region) const;

private:
  std::unique_ptr<TrackerDTC::Settings> settings_;
};

#endif