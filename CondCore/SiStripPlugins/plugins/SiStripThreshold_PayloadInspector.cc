/*!
  \file SiStripThreshold_PayloadInspector
  \Payload Inspector Plugin for SiStrip Threshold 
  \author J. Prisciandaro
  \version $Revision: 1.0 $
  \date $Date: 2018/02/22 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

using namespace std;

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    test class
  *************************************************/

  class SiStripThresholdTest : public Histogram1D<SiStripThreshold, SINGLE_IOV> {
  public:
    SiStripThresholdTest()
        : Histogram1D<SiStripThreshold, SINGLE_IOV>("SiStrip Threshold test", "SiStrip Threshold test", 10, 0.0, 10.0),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripThreshold> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          std::stringstream ss;
          ss << "Summary of strips threshold:" << std::endl;

          payload->printSummary(ss, &m_trackerTopo);

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          std::cout << ss.str() << std::endl;
        }
      }
      return true;
    }  // fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************************
    1d histogram of SiStripThresholds of 1 IOV - High Threshold 
  *************************************************************/

  class SiStripThresholdValueHigh : public Histogram1D<SiStripThreshold, SINGLE_IOV> {
  public:
    SiStripThresholdValueHigh()
        : Histogram1D<SiStripThreshold, SINGLE_IOV>("SiStrip High threshold values (checked per APV)",
                                                    "SiStrip High threshold values (cheched per APV)",
                                                    10,
                                                    0.0,
                                                    10) {}
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripThreshold> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            //std::cout<<d<<std::endl;
            SiStripThreshold::Range range = payload->getRange(d);

            int nAPVs = reader->getNumberOfApvsAndStripLength(d).first;

            for (int it = 0; it < nAPVs; ++it) {
              auto hth = payload->getData(it * 128, range).getHth();
              //std::cout<<hth<<std::endl;
              fillWithValue(hth);
            }
          }
        }
      }

      delete reader;
      return true;
    }
  };

  /************************************************************
    1d histogram of SiStripThresholds of 1 IOV - Low Threshold 
  *************************************************************/

  class SiStripThresholdValueLow : public Histogram1D<SiStripThreshold, SINGLE_IOV> {
  public:
    SiStripThresholdValueLow()
        : Histogram1D<SiStripThreshold, SINGLE_IOV>("SiStrip Low threshold values (checked per APV)",
                                                    "SiStrip Low threshold values (cheched per APV)",
                                                    10,
                                                    0.0,
                                                    10) {}
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripThreshold> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            //std::cout<<d<<std::endl;
            SiStripThreshold::Range range = payload->getRange(d);

            int nAPVs = reader->getNumberOfApvsAndStripLength(d).first;

            for (int it = 0; it < nAPVs; ++it) {
              auto lth = payload->getData(it * 128, range).getLth();
              //std::cout<<hth<<std::endl;
              fillWithValue(lth);
            }
          }
        }
      }

      delete reader;
      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripThreshold) {
  PAYLOAD_INSPECTOR_CLASS(SiStripThresholdTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripThresholdValueHigh);
  PAYLOAD_INSPECTOR_CLASS(SiStripThresholdValueLow);
}
