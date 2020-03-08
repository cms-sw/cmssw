#ifndef __L1TTrackerDTC_TTDTCCONVERTER_H__
#define __L1TTrackerDTC_TTDTCCONVERTER_H__

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "L1Trigger/L1TTrackerDTC/interface/Settings.h"

#include <string>
#include <memory>


// converts TTDTC::Frame into GlobalPoint
class TTDTCConverter {
public:
  TTDTCConverter(const edm::Run& iRun,
                 const edm::EventSetup& iSetup,
                 const std::string& processName = "",
                 const std::string& productLabel = "L1TTrackerDTCProducer");

  ~TTDTCConverter() { delete settings_; }

  // returns bit accurate position of a stub from a given processing region [0-8] (phi slice of outer tracker)
  GlobalPoint pos(const TTDTC::Frame& frame, const int& region) const;

private:
  L1TTrackerDTC::Settings* settings_;
};

#endif