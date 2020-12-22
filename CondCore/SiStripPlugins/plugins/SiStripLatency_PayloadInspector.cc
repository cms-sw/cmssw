/*!
  \file SiStripLatency_PayloadInspector
  \Payload Inspector Plugin for SiStrip Latency
  \author Jessica Prisciandaro
  \version $Revision: 1.0 $
  \date $Date: 2018/05/22 17:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TProfile.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
   ***************  test class ******************
  *************************************************/

  class SiStripLatencyTest : public Histogram1D<SiStripLatency, SINGLE_IOV> {
  public:
    SiStripLatencyTest()
        : Histogram1D<SiStripLatency, SINGLE_IOV>("SiStripLatency values", "SiStripLatency values", 5, 0.0, 5.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripLatency> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<SiStripLatency::Latency> lat = payload->allLatencyAndModes();
          fillWithValue(lat.size());
        }
      }
      return true;
    }  // fill
  };

  /***********************************************
  // 1d histogram of mode  of 1 IOV 
  ************************************************/
  class SiStripLatencyMode : public Histogram1D<SiStripLatency, SINGLE_IOV> {
  public:
    SiStripLatencyMode() : Histogram1D<SiStripLatency>("SiStripLatency mode", "SiStripLatency mode", 70, -10, 60) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripLatency> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint16_t> modes;
          payload->allModes(modes);

          for (const auto& mode : modes) {
            if (mode != 0)
              fillWithValue(mode);
            else
              fillWithValue(-1);
          }
        }
      }
      return true;
    }
  };

  /****************************************************************************
   *******************1D histo of mode as a function of the run****************
   *****************************************************************************/

  class SiStripLatencyModeHistory : public HistoryPlot<SiStripLatency, uint16_t> {
  public:
    SiStripLatencyModeHistory() : HistoryPlot<SiStripLatency, uint16_t>("Mode vs run number", "Mode vs run number") {}

    uint16_t getFromPayload(SiStripLatency& payload) override {
      uint16_t singlemode = payload.singleMode();
      return singlemode;
    }
  };

  /****************************************************************************    
   *****************number of modes  per run *************************************
   **************************************************************************/
  class SiStripLatencyNumbOfModeHistory : public HistoryPlot<SiStripLatency, int> {
  public:
    SiStripLatencyNumbOfModeHistory()
        : HistoryPlot<SiStripLatency, int>("Number of modes vs run ", "Number of modes vs run") {}

    int getFromPayload(SiStripLatency& payload) override {
      std::vector<uint16_t> modes;
      payload.allModes(modes);

      return modes.size();
    }
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripLatency) {
  PAYLOAD_INSPECTOR_CLASS(SiStripLatencyTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripLatencyMode);
  PAYLOAD_INSPECTOR_CLASS(SiStripLatencyModeHistory);
  PAYLOAD_INSPECTOR_CLASS(SiStripLatencyNumbOfModeHistory);
}
