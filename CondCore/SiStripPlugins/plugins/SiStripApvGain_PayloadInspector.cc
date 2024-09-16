/*!
  \file SiStripApvGains_PayloadInspector
  \Payload Inspector Plugin for SiStrip Gain
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/07/02 17:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "SiStripCondObjectRepresent.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TProfile.h"
#include "TH2F.h"
#include "THStack.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  class SiStripApvGainContainer : public SiStripCondObjectRepresent::SiStripDataContainer<SiStripApvGain, float> {
  public:
    SiStripApvGainContainer(const std::shared_ptr<SiStripApvGain>& payload,
                            const SiStripPI::MetaData& metadata,
                            const std::string& tagName)
        : SiStripCondObjectRepresent::SiStripDataContainer<SiStripApvGain, float>(payload, metadata, tagName) {
      payloadType_ = "SiStripApvGain";
      setGranularity(SiStripCondObjectRepresent::PERAPV);
    }

    void storeAllValues() override {
      std::vector<uint32_t> detid;
      payload_->getDetIds(detid);

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload_->getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          // to be used to fill the histogram
          SiStripCondData_.fillByPushBack(d, payload_->getApvGain(it, range));
        }
      }
    }
  };

  /************************************************
    testing the machinery
  ************************************************/
  class SiStripApvGainTest : public Histogram1D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainTest()
        : Histogram1D<SiStripApvGain, SINGLE_IOV>("SiStrip ApvGain values", "SiStrip ApvGain values", 1, 0.0, 1.) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto tagname = tag.name;
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          SiStripApvGainContainer* objContainer = new SiStripApvGainContainer(payload, iov, tagname);
          objContainer->printAll();

        }  // payload
      }  // iovs
      return true;
    }  // fill
  };

  class SiStripApvGainByPartition : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainByPartition() : PlotImage<SiStripApvGain, SINGLE_IOV>("SiStrip ApvGains By Partition") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiStripApvGain> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        SiStripApvGainContainer* objContainer = new SiStripApvGainContainer(payload, iov, tagname);
        //objContainer->printAll();

        TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
        objContainer->fillByPartition(canvas, 100, 0., 2.);

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }  // payload
      return true;
    }  // fill
  };

  class SiStripApvGainCompareByPartition : public PlotImage<SiStripApvGain, MULTI_IOV, 2> {
  public:
    SiStripApvGainCompareByPartition()
        : PlotImage<SiStripApvGain, MULTI_IOV, 2>("SiStrip Compare ApvGains By Partition") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      auto tag2iovs = PlotBase::getTag<1>().iovs;
      auto tagname2 = PlotBase::getTag<1>().name;
      SiStripPI::MetaData firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov = tag2iovs.front();

      std::shared_ptr<SiStripApvGain> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload(std::get<1>(firstiov));

      SiStripApvGainContainer* l_objContainer = new SiStripApvGainContainer(last_payload, lastiov, tagname1);
      SiStripApvGainContainer* f_objContainer = new SiStripApvGainContainer(first_payload, firstiov, tagname2);

      l_objContainer->compare(f_objContainer);

      //l_objContainer->printAll();

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      l_objContainer->fillByPartition(canvas, 100, 0.5, 1.5);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }  // fill
  };

  class SiStripApvGainRatioByPartition : public PlotImage<SiStripApvGain, MULTI_IOV, 2> {
  public:
    SiStripApvGainRatioByPartition() : PlotImage<SiStripApvGain, MULTI_IOV, 2>("SiStrip Ratio ApvGains By Partition") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      auto tag2iovs = PlotBase::getTag<1>().iovs;
      auto tagname2 = PlotBase::getTag<1>().name;
      SiStripPI::MetaData firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov = tag2iovs.front();

      std::shared_ptr<SiStripApvGain> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload(std::get<1>(firstiov));

      SiStripApvGainContainer* l_objContainer = new SiStripApvGainContainer(last_payload, lastiov, tagname1);
      SiStripApvGainContainer* f_objContainer = new SiStripApvGainContainer(first_payload, firstiov, tagname2);

      l_objContainer->divide(f_objContainer);

      //l_objContainer->printAll();

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      l_objContainer->fillByPartition(canvas, 200, 0.5, 1.5);
      //for (int i = 1; i <= 4; i++)
      //  canvas.cd(i)->SetLogy();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }  // fill
  };

  class SiStripApvGainDiffByPartition : public PlotImage<SiStripApvGain, MULTI_IOV, 2> {
  public:
    SiStripApvGainDiffByPartition() : PlotImage<SiStripApvGain, MULTI_IOV, 2>("SiStrip Diff ApvGains By Partition") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      auto tag2iovs = PlotBase::getTag<1>().iovs;
      auto tagname2 = PlotBase::getTag<1>().name;
      SiStripPI::MetaData firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov = tag2iovs.front();

      std::shared_ptr<SiStripApvGain> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload(std::get<1>(firstiov));

      SiStripApvGainContainer* l_objContainer = new SiStripApvGainContainer(last_payload, lastiov, tagname1);
      SiStripApvGainContainer* f_objContainer = new SiStripApvGainContainer(first_payload, firstiov, tagname2);

      l_objContainer->subtract(f_objContainer);

      //l_objContainer->printAll();

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      l_objContainer->fillByPartition(canvas, 100, -0.1, 0.1);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }  // fill
  };

  /************************************************
    1d histogram of SiStripApvGains of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvGainsValue : public Histogram1D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsValue()
        : Histogram1D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains values", "SiStripApv Gains values", 200, 0.0, 2.0) {
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              // to be used to fill the histogram
              fillWithValue(payload->getApvGain(it, range));

            }  // loop over APVs
          }  // loop over detIds
        }  // payload
      }  // iovs
      return true;
    }  // fill
  };

  /************************************************
    1d histogram of means of SiStripApvGains
    for Tracker Barrel of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvBarrelGainsByLayer : public Histogram1D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvBarrelGainsByLayer()
        : Histogram1D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains averages by Barrel layer",
                                                  "Barrel layer (0-3: TIB), (4-9: TOB)",
                                                  10,
                                                  0,
                                                  10,
                                                  "average SiStripApv Gain") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          std::map<int, std::pair<float, float>> sumOfGainsByLayer;

          for (const auto& d : detid) {
            int subid = DetId(d).subdetId();
            int layer(-1);
            if (subid != StripSubdetector::TIB && subid != StripSubdetector::TOB)
              continue;
            if (subid == StripSubdetector::TIB) {
              layer = tTopo.tibLayer(d);
            } else if (subid == StripSubdetector::TOB) {
              // layers of TOB start at 5th bin
              layer = tTopo.tobLayer(d);
              layer += 4;
            }

            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              sumOfGainsByLayer[layer].first += payload->getApvGain(it, range);
              sumOfGainsByLayer[layer].second += 1.;
            }  // loop over APVs
          }  // loop over detIds

          // loop on the map to fill the plot
          for (auto& data : sumOfGainsByLayer) {
            fillWithBinAndValue(data.first - 1, (data.second.first / data.second.second));
          }

        }  // payload
      }  // iovs
      return true;
    }  // fill
  };

  /************************************************
    2d histogram of absolute (i.e. not average)
    SiStripApvGains for Tracker Barrel of 1 IOV
  *************************************************/

  class SiStripApvAbsoluteBarrelGainsByLayer : public Histogram2D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvAbsoluteBarrelGainsByLayer()
        : Histogram2D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains by Barrel layer",
                                                  "Barrel layer (0-3: TIB), (4-9: TOB)",
                                                  10,
                                                  0,
                                                  10,
                                                  "SiStripApv Gain",
                                                  200,
                                                  0.0,
                                                  2.0) {}
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);
          for (const auto& d : detid) {
            int subid = DetId(d).subdetId();
            if (subid != 3 && subid != 5)
              continue;

            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              float gain = payload->getApvGain(it, range);
              fillWithValue(static_cast<float>((subid == 5) ? tTopo.tobLayer(d) + 3 : tTopo.tibLayer(d) - 1),
                            (gain > 2.0) ? 2.0 : gain);
            }
          }  //loop over detIds
        }  // loop over payloads
      }  // loop over iovs
      return true;
    }  // fill
  };

  /************************************************
    1d histogram of means of SiStripApvGains
    for Tracker Endcaps (minus side) of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvEndcapMinusGainsByDisk : public Histogram1D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvEndcapMinusGainsByDisk()
        : Histogram1D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains averages by Endcap (minus) disk",
                                                  "Endcap (minus) disk (0-2: TID), (3-11: TEC)",
                                                  12,
                                                  0,
                                                  12,
                                                  "average SiStripApv Gain") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          std::map<int, std::pair<float, float>> sumOfGainsByDisk;

          for (const auto& d : detid) {
            int disk = -1;
            int side = -1;
            int subid = DetId(d).subdetId();
            if (subid != StripSubdetector::TID && subid != StripSubdetector::TEC)
              continue;

            if (subid == StripSubdetector::TID) {
              side = tTopo.tidSide(d);
              disk = tTopo.tidWheel(d);
            } else {
              side = tTopo.tecSide(d);
              disk = tTopo.tecWheel(d);
              // disks of TEC start at 4th bin
              disk += 3;
            }

            // only negative side
            if (side != 1)
              continue;

            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              sumOfGainsByDisk[disk].first += payload->getApvGain(it, range);
              sumOfGainsByDisk[disk].second += 1.;
            }  // loop over APVs
          }  // loop over detIds

          // loop on the map to fill the plot
          for (auto& data : sumOfGainsByDisk) {
            fillWithBinAndValue(data.first - 1, (data.second.first / data.second.second));
          }

        }  // payload
      }  // iovs
      return true;
    }  // fill
  };

  /************************************************
    1d histogram of means of SiStripApvGains
    for Tracker Endcaps (plus side) of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvEndcapPlusGainsByDisk : public Histogram1D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvEndcapPlusGainsByDisk()
        : Histogram1D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains averages by Endcap (plus) disk",
                                                  "Endcap (plus) disk (0-2: TID), (3-11: TEC)",
                                                  12,
                                                  0,
                                                  12,
                                                  "average SiStripApv Gain") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          std::map<int, std::pair<float, float>> sumOfGainsByDisk;

          for (const auto& d : detid) {
            int disk = -1;
            int side = -1;
            int subid = DetId(d).subdetId();
            if (subid != StripSubdetector::TID && subid != StripSubdetector::TEC)
              continue;

            if (subid == StripSubdetector::TID) {
              side = tTopo.tidSide(d);
              disk = tTopo.tidWheel(d);
              ;
            } else {
              side = tTopo.tecSide(d);
              disk = tTopo.tecWheel(d);
              // disks of TEC start at 4th bin
              disk += 3;
            }

            // only positive side
            if (side != 2)
              continue;

            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              sumOfGainsByDisk[disk].first += payload->getApvGain(it, range);
              sumOfGainsByDisk[disk].second += 1.;
            }  // loop over APVs
          }  // loop over detIds

          // loop on the map to fill the plot
          for (auto& data : sumOfGainsByDisk) {
            fillWithBinAndValue(data.first - 1, (data.second.first / data.second.second));
          }

        }  // payload
      }  // iovs
      return true;
    }  // fill
  };

  /************************************************
    2D histogram of absolute (i.e. not average)
    SiStripApv Gains on the Endcap- for 1 IOV
   ************************************************/
  class SiStripApvAbsoluteEndcapMinusGainsByDisk : public Histogram2D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvAbsoluteEndcapMinusGainsByDisk()
        : Histogram2D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains averages by Endcap (minus) disk",
                                                  "Endcap (minus) disk (0-2: TID), (3-11: TEC)",
                                                  12,
                                                  0,
                                                  12,
                                                  "SiStripApv Gain",
                                                  200,
                                                  0.0,
                                                  2.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            int subid = DetId(d).subdetId(), side = -1, disk = -1;

            switch (subid) {
              case 4:
                side = tTopo.tidSide(d);
                disk = tTopo.tidWheel(d);
                break;
              case 6:
                side = tTopo.tecSide(d);
                disk = tTopo.tecWheel(d) + 4;
                break;
              default:
                continue;
            }

            if (side != 1)
              continue;
            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              float gain = payload->getApvGain(it, range);
              fillWithValue((float)disk - 1, (gain > 2.0) ? 2.0 : gain);
            }  // apvs
          }  // detids
        }
      }  // iovs
      return true;
    }  // fill
  };

  /************************************************
    2D histogram of absolute (i.e. not average)
    SiStripApv Gains on the Endcap+ for 1 IOV
   ************************************************/
  class SiStripApvAbsoluteEndcapPlusGainsByDisk : public Histogram2D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvAbsoluteEndcapPlusGainsByDisk()
        : Histogram2D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains averages by Endcap (plus) disk",
                                                  "Endcap (plus) disk (0-2: TID), (3-11: TEC)",
                                                  12,
                                                  0,
                                                  12,
                                                  "SiStripApv Gain",
                                                  200,
                                                  0.0,
                                                  2.0) {}
    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            int subid = DetId(d).subdetId(), side = -1, disk = -1;

            switch (subid) {
              case 4:
                side = tTopo.tidSide(d);
                disk = tTopo.tidWheel(d);
                break;
              case 6:
                side = tTopo.tecSide(d);
                disk = tTopo.tecWheel(d) + 4;
                break;
              default:
                continue;
            }

            if (side != 2)
              continue;
            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; it++) {
              float gain = payload->getApvGain(it, range);
              fillWithValue((float)disk - 1, (gain > 2.0) ? 2.0 : gain);
            }  //apvs
          }  //detids
        }
      }  // iovs
      return true;
    }  // fill
  };

  /************************************************
    TrackerMap of SiStripApvGains (average gain per detid)
  *************************************************/
  class SiStripApvGainsAverageTrackerMap : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsAverageTrackerMap() : PlotImage<SiStripApvGain, SINGLE_IOV>("Tracker Map of average SiStripGains") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload(std::get<1>(iov));

      std::string titleMap = "SiStrip APV Gain average per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripApvGains");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<uint32_t, float> store;

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload->getRange(d);
        float sumOfGains = 0;
        float nAPVsPerModule = 0.;
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVsPerModule += 1;
          sumOfGains += payload->getApvGain(it, range);
        }  // loop over APVs
        // fill the tracker map taking the average gain on a single DetId
        store[d] = (sumOfGains / nAPVsPerModule);
        tmap->fill(d, (sumOfGains / nAPVsPerModule));
      }  // loop over detIds

      //=========================
      // saturate at 2 std deviations
      auto range = SiStripPI::getTheRange(store, 2);

      std::string fileName(m_imageFileName);
      tmap->save(true, range.first, range.second, fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (module with default)
  *************************************************/
  class SiStripApvGainsDefaultTrackerMap : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsDefaultTrackerMap()
        : PlotImage<SiStripApvGain, SINGLE_IOV>("Tracker Map of SiStripGains to default") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload(std::get<1>(iov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripApvGains");

      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      /*
	the default G1 value comes from the ratio of DefaultTickHeight/GainNormalizationFactor
	as defined in the default of the O2O producer: OnlineDB/SiStripESSources/src/SiStripCondObjBuilderFromDb.cc
      */

      constexpr float G1default = 690. / 640.;
      constexpr float G2default = 1.f;

      int totalG1DefaultAPVs = 0;
      int totalG2DefaultAPVs = 0;

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload->getRange(d);
        float sumOfGains = 0;
        float nAPVsPerModule = 0.;
        int countDefaults = 0;
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVsPerModule += 1;
          sumOfGains += payload->getApvGain(it, range);
          if ((payload->getApvGain(it, range)) == G1default || (payload->getApvGain(it, range)) == G2default)
            countDefaults++;
        }  // loop over APVs
        // fill the tracker map taking the average gain on a single DetId
        if (countDefaults > 0.) {
          tmap->fill(d, countDefaults);
          // check here if it exists at least a full module with all APVs defaulted
          if (countDefaults == nAPVsPerModule) {
            if (std::fmod((sumOfGains / countDefaults), G1default) == 0.) {
              totalG1DefaultAPVs += countDefaults;
            } else if (std::fmod((sumOfGains / countDefaults), G2default) == 0.) {
              totalG2DefaultAPVs += countDefaults;
            }
          }
        }
      }  // loop over detIds

      //=========================
      std::string gainType{""};
      if (totalG2DefaultAPVs == 0) {
        gainType = "G1 value (=690./640.)";
      } else if (totalG1DefaultAPVs == 0) {
        gainType = "G2 value (=1)";
      }

      std::string titleMap = "# of APVs/module w/ default " + gainType + " (payload : " + std::get<1>(iov) + ")";
      tmap->setTitle(titleMap);

      std::string fileName(m_imageFileName);
      tmap->save(true, 0, 0, fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (ratio with previous gain per detid)
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripApvGainsRatioTrackerMapBase : public PlotImage<SiStripApvGain, nIOVs, ntags> {
  public:
    SiStripApvGainsRatioTrackerMapBase()
        : PlotImage<SiStripApvGain, nIOVs, ntags>("Tracker Map of ratio of SiStripGains with previous IOV") {
      PlotBase::addInputParam("nsigma");
    }

    bool fill() override {
      // determine n. sigmas
      unsigned int nsigma(1);

      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("nsigma");
      if (ip != paramValues.end()) {
        nsigma = std::stoul(ip->second);
      }

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripApvGain> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string titleMap = "SiStrip APV Gain ratio per module average (IOV: ";

      titleMap += std::to_string(std::get<0>(firstiov));
      titleMap += "/ IOV:";
      titleMap += std::to_string(std::get<0>(lastiov));
      titleMap += ")";

      titleMap += +" " + std::to_string(nsigma) + " std. dev. saturation";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripApvGains");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::map<uint32_t, float> lastmap, firstmap;

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      // cache the last IOV
      for (const auto& d : detid) {
        SiStripApvGain::Range range = last_payload->getRange(d);
        float Gain = 0;
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; it++) {
          nAPV += 1;
          Gain += last_payload->getApvGain(it, range);
        }  // loop over APVs
        lastmap[d] = (Gain / nAPV);
      }  // loop over detIds

      detid.clear();

      first_payload->getDetIds(detid);

      // cache the first IOV
      for (const auto& d : detid) {
        SiStripApvGain::Range range = first_payload->getRange(d);
        float Gain = 0;
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; it++) {
          nAPV += 1;
          Gain += first_payload->getApvGain(it, range);
        }  // loop over APVs
        firstmap[d] = (Gain / nAPV);
      }  // loop over detIds

      std::map<uint32_t, float> cachedRatio;
      for (const auto& d : detid) {
        float ratio = firstmap[d] / lastmap[d];
        tmap->fill(d, ratio);
        cachedRatio[d] = ratio;
      }

      //=========================
      auto range = SiStripPI::getTheRange(cachedRatio, nsigma);

      std::string fileName(this->m_imageFileName);
      tmap->save(true, range.first, range.second, fileName);

      return true;
    }
  };

  using SiStripApvGainsAvgDeviationRatioWithPreviousIOVTrackerMap = SiStripApvGainsRatioTrackerMapBase<1, MULTI_IOV>;
  using SiStripApvGainsAvgDeviationRatioTrackerMapTwoTags = SiStripApvGainsRatioTrackerMapBase<2, SINGLE_IOV>;

  /************************************************
   TrackerMap of SiStripApvGains (ratio for largest deviation with previous gain per detid)
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripApvGainsRatioMaxDeviationTrackerMapBase : public PlotImage<SiStripApvGain, nIOVs, ntags> {
  public:
    SiStripApvGainsRatioMaxDeviationTrackerMapBase()
        : PlotImage<SiStripApvGain, nIOVs, ntags>(
              "Tracker Map of ratio (for largest deviation) of SiStripGains with previous IOV") {
      PlotBase::addInputParam("nsigma");
    }

    bool fill() override {
      unsigned int nsigma(1);
      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("nsigma");
      if (ip != paramValues.end()) {
        nsigma = std::stoul(ip->second);
        edm::LogPrint("SiStripApvGain_PayloadInspector")
            << "using custom z-axis saturation: " << nsigma << " sigmas" << std::endl;
      } else {
        edm::LogPrint("SiStripApvGain_PayloadInspector")
            << "using default saturation: " << nsigma << " sigmas" << std::endl;
      }

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripApvGain> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string titleMap = "SiStrip APV Gain ratio for largest deviation per module (IOV: ";

      titleMap += std::to_string(std::get<0>(firstiov));
      titleMap += "/ IOV:";
      titleMap += std::to_string(std::get<0>(lastiov));
      titleMap += ") ";

      titleMap += +" - " + std::to_string(nsigma) + " std. dev. saturation";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripApvGains");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::map<std::pair<uint32_t, int>, float> lastmap, firstmap;

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      // cache the last IOV
      for (const auto& d : detid) {
        SiStripApvGain::Range range = last_payload->getRange(d);
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; it++) {
          nAPV += 1;
          float Gain = last_payload->getApvGain(it, range);
          std::pair<uint32_t, int> index = std::make_pair(d, nAPV);
          lastmap[index] = Gain;
        }  // loop over APVs
      }  // loop over detIds

      detid.clear();

      first_payload->getDetIds(detid);

      // cache the first IOV
      for (const auto& d : detid) {
        SiStripApvGain::Range range = first_payload->getRange(d);
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; it++) {
          nAPV += 1;
          float Gain = first_payload->getApvGain(it, range);
          std::pair<uint32_t, int> index = std::make_pair(d, nAPV);
          firstmap[index] = Gain;
        }  // loop over APVs
      }  // loop over detIds

      // find the largest deviation
      std::map<uint32_t, float> cachedRatio;

      for (const auto& item : firstmap) {
        // packed index (detid,APV)
        auto index = item.first;
        auto mod = item.first.first;

        float ratio = firstmap[index] / lastmap[index];
        // if we have already cached something
        if (cachedRatio[mod]) {
          // if the discrepancy with 1 of ratio is larger than the cached value
          if (std::abs(ratio - 1.) > std::abs(cachedRatio[mod] - 1.)) {
            cachedRatio[mod] = ratio;
          }
        } else {
          cachedRatio[mod] = ratio;
        }
      }

      for (const auto& element : cachedRatio) {
        tmap->fill(element.first, element.second);
      }

      // get the range of the TrackerMap (saturate at +/-n std deviations)
      auto range = SiStripPI::getTheRange(cachedRatio, nsigma);

      //=========================

      std::string fileName(this->m_imageFileName);
      tmap->save(true, range.first, range.second, fileName);

      return true;
    }
  };

  using SiStripApvGainsMaxDeviationRatioWithPreviousIOVTrackerMap =
      SiStripApvGainsRatioMaxDeviationTrackerMapBase<1, MULTI_IOV>;

  using SiStripApvGainsMaxDeviationRatioTrackerMapTwoTags =
      SiStripApvGainsRatioMaxDeviationTrackerMapBase<2, SINGLE_IOV>;

  /************************************************
    TrackerMap of SiStripApvGains (maximum gain per detid)
  *************************************************/
  class SiStripApvGainsMaximumTrackerMap : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsMaximumTrackerMap()
        : PlotImage<SiStripApvGain, SINGLE_IOV>("Tracker Map of SiStripAPVGains (maximum per DetId)") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload(std::get<1>(iov));

      std::string titleMap = "SiStrip APV Gain maximum per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripApvGains");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload->getRange(d);
        float theMaxGain = 0;
        for (int it = 0; it < range.second - range.first; it++) {
          float currentGain = payload->getApvGain(it, range);
          if (currentGain > theMaxGain) {
            theMaxGain = currentGain;
          }
        }  // loop over APVs
        // fill the tracker map taking the maximum gain on a single DetId
        tmap->fill(d, theMaxGain);
      }  // loop over detIds

      //=========================

      std::pair<float, float> extrema = tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);

      // protect against uniform values (gains are defined positive)
      if (extrema.first != extrema.second) {
        tmap->save(true, 0, 0, fileName);
      } else {
        tmap->save(true, extrema.first * 0.95, extrema.first * 1.05, fileName);
      }

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (minimum gain per detid)
  *************************************************/
  class SiStripApvGainsMinimumTrackerMap : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsMinimumTrackerMap()
        : PlotImage<SiStripApvGain, SINGLE_IOV>("Tracker Map of SiStripAPVGains (minimum per DetId)") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<SiStripApvGain> payload = fetchPayload(std::get<1>(iov));

      std::string titleMap = "SiStrip APV Gain minumum per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripApvGains");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload->getRange(d);
        float theMinGain = 999.;
        for (int it = 0; it < range.second - range.first; it++) {
          float currentGain = payload->getApvGain(it, range);
          if (currentGain < theMinGain) {
            theMinGain = currentGain;
          }
        }  // loop over APVs
        // fill the tracker map taking the minimum gain on a single DetId
        tmap->fill(d, theMinGain);
      }  // loop over detIds

      //=========================

      std::pair<float, float> extrema = tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);

      // protect against uniform values (gains are defined positive)
      if (extrema.first != extrema.second) {
        tmap->save(true, 0, 0, fileName);
      } else {
        tmap->save(true, extrema.first * 0.95, extrema.first * 1.05, fileName);
      }

      return true;
    }
  };

  /************************************************
    time history histogram of SiStripApvGains 
  *************************************************/

  class SiStripApvGainByRunMeans : public HistoryPlot<SiStripApvGain, float> {
  public:
    SiStripApvGainByRunMeans()
        : HistoryPlot<SiStripApvGain, float>("SiStripApv Gains average", "average Strip APV gain value") {}
    ~SiStripApvGainByRunMeans() override = default;

    float getFromPayload(SiStripApvGain& payload) override {
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      float nAPVs = 0;
      float sumOfGains = 0;

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload.getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVs += 1;
          sumOfGains += payload.getApvGain(it, range);
        }  // loop over APVs
      }  // loop over detIds

      return sumOfGains / nAPVs;
    }  // payload
  };

  /************************************************
    time history of SiStripApvGains properties
  *************************************************/

  template <SiStripPI::estimator est>
  class SiStripApvGainProperties : public HistoryPlot<SiStripApvGain, float> {
  public:
    SiStripApvGainProperties()
        : HistoryPlot<SiStripApvGain, float>("SiStripApv Gains " + estimatorType(est),
                                             estimatorType(est) + " Strip APV gain value") {}
    ~SiStripApvGainProperties() override = default;

    float getFromPayload(SiStripApvGain& payload) override {
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      float nAPVs = 0;
      float sumOfGains = 0;
      float meanOfGains = 0;
      float rmsOfGains = 0;
      float min(0.), max(0.);

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload.getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVs += 1;
          float gain = payload.getApvGain(it, range);
          if (gain < min)
            min = gain;
          if (gain > max)
            max = gain;
          sumOfGains += gain;
          rmsOfGains += (gain * gain);
        }  // loop over APVs
      }  // loop over detIds

      meanOfGains = sumOfGains / nAPVs;

      switch (est) {
        case SiStripPI::min:
          return min;
          break;
        case SiStripPI::max:
          return max;
          break;
        case SiStripPI::mean:
          return meanOfGains;
          break;
        case SiStripPI::rms:
          if ((rmsOfGains / nAPVs - meanOfGains * meanOfGains) > 0.) {
            return sqrt(rmsOfGains / nAPVs - meanOfGains * meanOfGains);
          } else {
            return 0.;
          }
          break;
        default:
          edm::LogWarning("LogicError") << "Unknown estimator: " << est;
          break;
      }
      return 0.;
    }  // payload
  };

  typedef SiStripApvGainProperties<SiStripPI::min> SiStripApvGainMin_History;
  typedef SiStripApvGainProperties<SiStripPI::max> SiStripApvGainMax_History;
  typedef SiStripApvGainProperties<SiStripPI::mean> SiStripApvGainMean_History;
  typedef SiStripApvGainProperties<SiStripPI::rms> SiStripApvGainRMS_History;

  /************************************************
    time history histogram of TIB SiStripApvGains 
  *************************************************/

  class SiStripApvTIBGainByRunMeans : public HistoryPlot<SiStripApvGain, float> {
  public:
    SiStripApvTIBGainByRunMeans()
        : HistoryPlot<SiStripApvGain, float>("SiStripApv Gains average",
                                             "average Tracker Inner Barrel APV gain value") {}
    ~SiStripApvTIBGainByRunMeans() override = default;

    float getFromPayload(SiStripApvGain& payload) override {
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      float nAPVs = 0;
      float sumOfGains = 0;

      for (const auto& d : detid) {
        int subid = DetId(d).subdetId();
        if (subid != StripSubdetector::TIB)
          continue;

        SiStripApvGain::Range range = payload.getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVs += 1;
          sumOfGains += payload.getApvGain(it, range);
        }  // loop over APVs
      }  // loop over detIds

      return sumOfGains / nAPVs;

    }  // payload
  };

  /************************************************
    time history histogram of TOB SiStripApvGains 
  *************************************************/

  class SiStripApvTOBGainByRunMeans : public HistoryPlot<SiStripApvGain, float> {
  public:
    SiStripApvTOBGainByRunMeans()
        : HistoryPlot<SiStripApvGain, float>("SiStripApv Gains average", "average Tracker Outer Barrel gain value") {}
    ~SiStripApvTOBGainByRunMeans() override = default;

    float getFromPayload(SiStripApvGain& payload) override {
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      float nAPVs = 0;
      float sumOfGains = 0;

      for (const auto& d : detid) {
        int subid = DetId(d).subdetId();
        if (subid != StripSubdetector::TOB)
          continue;

        SiStripApvGain::Range range = payload.getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVs += 1;
          sumOfGains += payload.getApvGain(it, range);
        }  // loop over APVs
      }  // loop over detIds

      return sumOfGains / nAPVs;

    }  // payload
  };

  /************************************************
    time history histogram of TID SiStripApvGains 
  *************************************************/

  class SiStripApvTIDGainByRunMeans : public HistoryPlot<SiStripApvGain, float> {
  public:
    SiStripApvTIDGainByRunMeans()
        : HistoryPlot<SiStripApvGain, float>("SiStripApv Gains average", "average Tracker Inner Disks APV gain value") {
    }
    ~SiStripApvTIDGainByRunMeans() override = default;

    float getFromPayload(SiStripApvGain& payload) override {
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      float nAPVs = 0;
      float sumOfGains = 0;
      for (const auto& d : detid) {
        int subid = DetId(d).subdetId();
        if (subid != StripSubdetector::TID)
          continue;

        SiStripApvGain::Range range = payload.getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVs += 1;
          sumOfGains += payload.getApvGain(it, range);
        }  // loop over APVs
      }  // loop over detIds

      return sumOfGains / nAPVs;

    }  // payload
  };

  /************************************************
    time history histogram of TEC SiStripApvGains 
  *************************************************/

  class SiStripApvTECGainByRunMeans : public HistoryPlot<SiStripApvGain, float> {
  public:
    SiStripApvTECGainByRunMeans()
        : HistoryPlot<SiStripApvGain, float>("SiStripApv Gains average in TEC",
                                             "average Tracker Endcaps APV gain value") {}
    ~SiStripApvTECGainByRunMeans() override = default;

    float getFromPayload(SiStripApvGain& payload) override {
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      float nAPVs = 0;
      float sumOfGains = 0;

      for (const auto& d : detid) {
        int subid = DetId(d).subdetId();
        if (subid != StripSubdetector::TEC)
          continue;

        SiStripApvGain::Range range = payload.getRange(d);
        for (int it = 0; it < range.second - range.first; it++) {
          nAPVs += 1;
          sumOfGains += payload.getApvGain(it, range);
        }  // loop over APVs
      }  // loop over detIds

      return sumOfGains / nAPVs;

    }  // payload
  };

  /************************************************
    test class
  *************************************************/

  class SiStripApvGainsTest : public Histogram1D<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsTest()
        : Histogram1D<SiStripApvGain, SINGLE_IOV>("SiStripApv Gains test", "SiStripApv Gains test", 10, 0.0, 10.0),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          SiStripDetSummary summaryGain{&m_trackerTopo};

          for (const auto& d : detid) {
            SiStripApvGain::Range range = payload->getRange(d);
            for (int it = 0; it < range.second - range.first; ++it) {
              summaryGain.add(d, payload->getApvGain(it, range));
              fillWithValue(payload->getApvGain(it, range));
            }
          }
          std::map<unsigned int, SiStripDetSummary::Values> map = summaryGain.getCounts();

          //SiStripPI::printSummary(map);

          std::stringstream ss;
          ss << "Summary of gain values:" << std::endl;
          summaryGain.print(ss, true);
          std::cout << ss.str() << std::endl;

        }  // payload
      }  // iovs
      return true;
    }  // fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Compare Gains from 2 IOVs, 2 pads canvas, firsr for ratio, second for scatter plot
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripApvGainsComparatorBase : public PlotImage<SiStripApvGain, nIOVs, ntags> {
  public:
    SiStripApvGainsComparatorBase() : PlotImage<SiStripApvGain, nIOVs, ntags>("SiStripGains Comparison") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripApvGain> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      std::map<std::pair<uint32_t, int>, float> lastmap, firstmap;

      // loop on the last payload
      for (const auto& d : detid) {
        SiStripApvGain::Range range = last_payload->getRange(d);
        float Gain = 0;
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; ++it) {
          nAPV += 1;
          Gain = last_payload->getApvGain(it, range);
          std::pair<uint32_t, int> index = std::make_pair(d, nAPV);
          lastmap[index] = Gain;
        }  // end loop on APVs
      }  // end loop on detids

      detid.clear();
      first_payload->getDetIds(detid);

      // loop on the first payload
      for (const auto& d : detid) {
        SiStripApvGain::Range range = first_payload->getRange(d);
        float Gain = 0;
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; ++it) {
          nAPV += 1;
          Gain = first_payload->getApvGain(it, range);
          std::pair<uint32_t, int> index = std::make_pair(d, nAPV);
          firstmap[index] = Gain;
        }  // end loop on APVs
      }  // end loop on detids

      TCanvas canvas("Payload comparison", "payload comparison", 1400, 1000);
      canvas.Divide(2, 1);

      std::map<std::string, std::shared_ptr<TH1F>> ratios;
      std::map<std::string, std::shared_ptr<TH2F>> scatters;
      std::map<std::string, int> colormap;
      std::map<std::string, int> markermap;
      colormap["TIB"] = kRed;
      markermap["TIB"] = kFullCircle;
      colormap["TOB"] = kGreen;
      markermap["TOB"] = kFullTriangleUp;
      colormap["TID"] = kBlack;
      markermap["TID"] = kFullSquare;
      colormap["TEC"] = kBlue;
      markermap["TEC"] = kFullTriangleDown;

      std::vector<std::string> parts = {"TEC", "TOB", "TIB", "TID"};

      for (const auto& part : parts) {
        ratios[part] = std::make_shared<TH1F>(
            Form("hRatio_%s", part.c_str()),
            Form("Gains ratio IOV: %s/ IOV: %s ;Previous Gain (%s) / New Gain (%s);Number of APV",
                 firstIOVsince.c_str(),
                 lastIOVsince.c_str(),
                 firstIOVsince.c_str(),
                 lastIOVsince.c_str()),
            200,
            0.,
            2.);
        scatters[part] =
            std::make_shared<TH2F>(Form("hScatter_%s", part.c_str()),
                                   Form("new Gain (%s) vs previous Gain (%s);Previous Gain (%s);New Gain (%s)",
                                        lastIOVsince.c_str(),
                                        firstIOVsince.c_str(),
                                        firstIOVsince.c_str(),
                                        lastIOVsince.c_str()),
                                   100,
                                   0.5,
                                   1.8,
                                   100,
                                   0.5,
                                   1.8);
      }

      // now loop on the cached maps
      for (const auto& item : firstmap) {
        // packed index (detid,APV)
        auto index = item.first;
        auto mod = item.first.first;

        int subid = DetId(mod).subdetId();
        float ratio = firstmap[index] / lastmap[index];

        if (subid == StripSubdetector::TIB) {
          ratios["TIB"]->Fill(ratio);
          scatters["TIB"]->Fill(firstmap[index], lastmap[index]);
        }

        if (subid == StripSubdetector::TOB) {
          ratios["TOB"]->Fill(ratio);
          scatters["TOB"]->Fill(firstmap[index], lastmap[index]);
        }

        if (subid == StripSubdetector::TID) {
          ratios["TID"]->Fill(ratio);
          scatters["TID"]->Fill(firstmap[index], lastmap[index]);
        }

        if (subid == StripSubdetector::TEC) {
          ratios["TEC"]->Fill(ratio);
          scatters["TEC"]->Fill(firstmap[index], lastmap[index]);
        }
      }

      auto legend = TLegend(0.60, 0.8, 0.92, 0.95);
      legend.SetTextSize(0.05);
      canvas.cd(1)->SetLogy();
      canvas.cd(1)->SetTopMargin(0.05);
      canvas.cd(1)->SetLeftMargin(0.13);
      canvas.cd(1)->SetRightMargin(0.08);

      for (const auto& part : parts) {
        SiStripPI::makeNicePlotStyle(ratios[part].get());
        ratios[part]->SetMinimum(1.);
        ratios[part]->SetStats(false);
        ratios[part]->SetLineWidth(2);
        ratios[part]->SetLineColor(colormap[part]);
        if (part == "TEC")
          ratios[part]->Draw();
        else
          ratios[part]->Draw("same");
        legend.AddEntry(ratios[part].get(), part.c_str(), "L");
      }

      legend.Draw("same");
      SiStripPI::drawStatBox(ratios, colormap, parts);

      auto legend2 = TLegend(0.60, 0.8, 0.92, 0.95);
      legend2.SetTextSize(0.05);
      canvas.cd(2);
      canvas.cd(2)->SetTopMargin(0.05);
      canvas.cd(2)->SetLeftMargin(0.13);
      canvas.cd(2)->SetRightMargin(0.08);

      for (const auto& part : parts) {
        SiStripPI::makeNicePlotStyle(scatters[part].get());
        scatters[part]->SetStats(false);
        scatters[part]->SetMarkerColor(colormap[part]);
        scatters[part]->SetMarkerStyle(markermap[part]);
        scatters[part]->SetMarkerSize(0.5);

        auto temp = (TH2F*)(scatters[part]->Clone());
        temp->SetMarkerSize(1.3);

        if (part == "TEC")
          scatters[part]->Draw("P");
        else
          scatters[part]->Draw("Psame");

        legend2.AddEntry(temp, part.c_str(), "P");
      }

      TLine diagonal(0.5, 0.5, 1.8, 1.8);
      diagonal.SetLineWidth(3);
      diagonal.SetLineStyle(2);
      diagonal.Draw("same");

      legend2.Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  using SiStripApvGainsComparatorSingleTag = SiStripApvGainsComparatorBase<1, MULTI_IOV>;
  using SiStripApvGainsComparatorTwoTags = SiStripApvGainsComparatorBase<2, SINGLE_IOV>;

  /************************************************
    Plot stack of gain by region
  *************************************************/

  class SiStripApvGainsTHStack : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsTHStack()
        : PlotImage<SiStripApvGain, SINGLE_IOV>("Stack of SiStrip APV gains values"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      //TH1F::SetDefaultSumw2(true);
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<SiStripApvGain> payload = this->fetchPayload(std::get<1>(iov));
      std::string IOVsince = std::to_string(std::get<0>(iov));

      std::map<partition, std::shared_ptr<TH1F>> h_gains;

      //std::vector< SiStripPI::TrackerRegion > regions;
      std::vector<partition> regions;

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      // fill the vector of regions
      for (const auto& d : detid) {
        //auto region = this->getTheRegion(d);
        auto region = this->getThePartition(d);
        if (std::find(regions.begin(), regions.end(), region) == regions.end()) {
          regions.push_back(region);
        }
      }

      LogDebug("SiStripApvGainsTHStack") << "regions.size()=" << regions.size() << std::endl;

      for (const auto& r : regions) {
        //auto part = std::string(SiStripPI::regionType(r).second);

        auto part = std::string(this->partitionName(r));

        h_gains[r] = std::make_shared<TH1F>(Form("hGains_%s", part.c_str()),
                                            Form("Gains values for IOV: %s ;Gain;Number of APV", IOVsince.c_str()),
                                            100,
                                            0.5,
                                            1.5);
      }

      // loop on the payload
      for (const auto& d : detid) {
        //auto region = this->getTheRegion(d);
        auto region = this->getThePartition(d);
        SiStripApvGain::Range range = payload->getRange(d);
        for (int it = 0; it < range.second - range.first; ++it) {
          float gain = payload->getApvGain(it, range);
          h_gains[region]->Fill(gain);
        }  // end loop on APVs
      }  // end loop on detids

      TCanvas canvas("Payload breakout", "payload breakout", 1200, 800);
      canvas.Divide(2, 1);

      std::array<int, 6> colors = {{kRed, kBlue, kGreen, kCyan, 8, kMagenta}};

      THStack* hs = new THStack("hs", Form("Gains values for IOV: %s;Gain;Number of APV", IOVsince.c_str()));
      int colorCounter = 0;
      for (const auto& r : regions) {
        hs->Add(h_gains[r].get());
        SiStripPI::makeNicePlotStyle(h_gains[r].get());
        h_gains[r]->SetFillColor(colors[colorCounter]);
        //h_gains[r]->SetLineColor(colorCounter);
        h_gains[r]->SetLineWidth(2);
        colorCounter++;
      }

      TLegend legend = TLegend(0.60, 0.65, 0.95, 0.93);
      legend.SetTextSize(0.05);
      legend.SetHeader("Gain break-out", "C");  // option "C" allows to center the header
      for (const auto& r : regions) {
        auto part = std::string(this->partitionName(r));
        legend.AddEntry(h_gains[r].get(), part.c_str(), "F");
      }

      canvas.cd(1)->SetLogy();
      canvas.cd(1)->SetTopMargin(0.07);
      canvas.cd(1)->SetBottomMargin(0.10);
      canvas.cd(1)->SetLeftMargin(0.15);
      canvas.cd(1)->SetRightMargin(0.05);
      //      hs->Draw("NOSTACKB");

      int count(0);
      auto stack = hs->GetHists();
      double maximum = hs->GetMaximum("nostack");  //SiStripPI::getMaximum(stack);

      TLegend legend2 = TLegend(0.70, 0.65, 0.95, 0.93);
      legend2.SetTextSize(0.05);
      legend2.SetHeader("Partition", "C");  // option "C" allows to center the header

      for (const auto&& elem : *stack) {
        auto clone = (TH1F*)(elem->Clone(Form("hclone_%s", elem->GetName())));
        SiStripPI::makeNicePlotStyle(clone);
        clone->SetFillColor(0);
        clone->SetMarkerStyle(20);
        clone->SetLineColor(colors[count]);
        clone->SetMarkerColor(colors[count]);
        clone->SetMaximum(maximum * 10);
        TString candName = clone->GetName();
        legend2.AddEntry(clone, candName.ReplaceAll("hclone_hGains_", ""), "L");
        if (count == 0) {
          clone->Draw("HIST");
        } else {
          clone->Draw("HISTsame");
        }
        count++;
      }

      legend2.Draw("same");

      canvas.cd(2);  //->SetLogy();
      canvas.cd(2)->SetTopMargin(0.07);
      canvas.cd(2)->SetBottomMargin(0.10);
      canvas.cd(2)->SetLeftMargin(0.12);
      canvas.cd(2)->SetRightMargin(0.05);
      hs->Draw();
      // all graphics manipulations *after* drawing the stack!
      hs->GetYaxis()->SetMaxDigits(2);
      SiStripPI::makeNiceStyle<THStack>(hs);
      legend.Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
    enum partition { TIB = 30, TIDP = 41, TIDM = 42, TOB = 50, TECP = 61, TECM = 62, END_OF_PARTS };

    const char* partitionName(partition part) {
      std::map<partition, const char*> mapping = {{partition::TIB, "TIB"},
                                                  {partition::TIDP, "TIPp"},
                                                  {partition::TIDM, "TIDm"},
                                                  {partition::TOB, "TOB"},
                                                  {partition::TECP, "TECp"},
                                                  {partition::TECM, "TECm"}};

      if (mapping.find(part) == mapping.end()) {
        throw cms::Exception("Invalid Partition passed");
      } else {
        return mapping[part];
      }
    }

    partition getThePartition(DetId detid) {
      int detNum = 0;
      int side = 0;
      switch (detid.subdetId()) {
        case StripSubdetector::TIB:
          detNum = 30;
          break;
        case StripSubdetector::TOB:
          detNum = 50;
          break;
        case StripSubdetector::TEC:
          // is this module in TEC+ or TEC-?
          side = m_trackerTopo.tecSide(detid);
          detNum = 60;
          break;
        case StripSubdetector::TID:
          // is this module in TID+ or TID-?
          side = m_trackerTopo.tidSide(detid);
          detNum = 40;
          break;
      }

      detNum += side;
      return static_cast<partition>(detNum);
    }

    SiStripPI::TrackerRegion getTheRegion(DetId detid) {
      int layer = 0;
      int stereo = 0;
      int detNum = 0;

      switch (detid.subdetId()) {
        case StripSubdetector::TIB:
          layer = m_trackerTopo.tibLayer(detid);
          stereo = m_trackerTopo.tibStereo(detid);
          detNum = 1000;
          break;
        case StripSubdetector::TOB:
          layer = m_trackerTopo.tobLayer(detid);
          stereo = m_trackerTopo.tobStereo(detid);
          detNum = 2000;
          break;
        case StripSubdetector::TEC:
          // is this module in TEC+ or TEC-?
          layer = m_trackerTopo.tecWheel(detid);
          stereo = m_trackerTopo.tecStereo(detid);
          detNum = 3000;
          break;
        case StripSubdetector::TID:
          // is this module in TID+ or TID-?
          layer = m_trackerTopo.tidWheel(detid);
          stereo = m_trackerTopo.tidStereo(detid);
          detNum = 4000;
          break;
      }

      detNum += layer * 10 + stereo;
      return static_cast<SiStripPI::TrackerRegion>(detNum);
    }
  };

  //*******************************************//
  // Compare Gains from 2 IOVs
  //******************************************//

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripApvGainsValuesComparatorBase : public PlotImage<SiStripApvGain, nIOVs, ntags> {
  public:
    SiStripApvGainsValuesComparatorBase()
        : PlotImage<SiStripApvGain, nIOVs, ntags>("Comparison of SiStrip APV gains values") {}

    bool fill() override {
      TH1F::SetDefaultSumw2(true);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripApvGain> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      std::map<std::pair<uint32_t, int>, float> lastmap, firstmap;

      // loop on the last payload
      for (const auto& d : detid) {
        SiStripApvGain::Range range = last_payload->getRange(d);
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; ++it) {
          nAPV += 1;
          auto index = std::make_pair(d, nAPV);
          lastmap[index] = last_payload->getApvGain(it, range);
        }  // end loop on APVs
      }  // end loop on detids

      detid.clear();
      first_payload->getDetIds(detid);

      // loop on the first payload
      for (const auto& d : detid) {
        SiStripApvGain::Range range = first_payload->getRange(d);
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; ++it) {
          nAPV += 1;
          auto index = std::make_pair(d, nAPV);
          firstmap[index] = last_payload->getApvGain(it, range);
        }  // end loop on APVs
      }  // end loop on detids

      TCanvas canvas("Payload comparison", "payload comparison", 1000, 1000);
      canvas.cd();

      TPad pad1("pad1", "pad1", 0, 0.3, 1, 1.0);
      pad1.SetBottomMargin(0.02);  // Upper and lower plot are joined
      pad1.SetTopMargin(0.07);
      pad1.SetRightMargin(0.05);
      pad1.SetLeftMargin(0.15);
      pad1.Draw();  // Draw the upper pad: pad1
      pad1.cd();    // pad1 becomes the current pad

      auto h_firstGains =
          std::make_shared<TH1F>("hFirstGains", "SiStrip APV gains values; APV Gains;n. APVs", 200, 0.2, 1.8);
      auto h_lastGains =
          std::make_shared<TH1F>("hLastGains", "SiStrip APV gains values; APV Gains;n. APVs", 200, 0.2, 1.8);

      for (const auto& item : firstmap) {
        h_firstGains->Fill(item.second);
      }

      for (const auto& item : lastmap) {
        h_lastGains->Fill(item.second);
      }

      SiStripPI::makeNicePlotStyle(h_lastGains.get());
      SiStripPI::makeNicePlotStyle(h_firstGains.get());

      TH1F* hratio = (TH1F*)h_firstGains->Clone("hratio");

      h_firstGains->SetLineColor(kRed);
      h_lastGains->SetLineColor(kBlue);

      h_firstGains->SetMarkerColor(kRed);
      h_lastGains->SetMarkerColor(kBlue);

      h_firstGains->SetMarkerSize(1.);
      h_lastGains->SetMarkerSize(1.);

      h_firstGains->SetLineWidth(1);
      h_lastGains->SetLineWidth(1);

      h_firstGains->SetMarkerStyle(20);
      h_lastGains->SetMarkerStyle(21);

      h_firstGains->GetXaxis()->SetLabelOffset(2.);
      h_lastGains->GetXaxis()->SetLabelOffset(2.);

      h_firstGains->Draw("HIST");
      h_lastGains->Draw("HISTsame");

      TLegend legend = TLegend(0.70, 0.7, 0.95, 0.9);
      legend.SetHeader("Gain Comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(h_firstGains.get(), ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "PL");
      legend.AddEntry(h_lastGains.get(), ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "PL");
      legend.Draw("same");

      // lower plot will be in pad
      canvas.cd();  // Go back to the main canvas before defining pad2
      TPad pad2("pad2", "pad2", 0, 0.005, 1, 0.3);
      pad2.SetTopMargin(0.01);
      pad2.SetBottomMargin(0.2);
      pad2.SetRightMargin(0.05);
      pad2.SetLeftMargin(0.15);
      pad2.SetGridy();  // horizontal grid
      pad2.Draw();
      pad2.cd();  // pad2 becomes the current pad

      // Define the ratio plot
      hratio->SetLineColor(kBlack);
      hratio->SetMarkerColor(kBlack);
      hratio->SetTitle("");
      hratio->SetMinimum(0.55);  // Define Y ..
      hratio->SetMaximum(1.55);  // .. range
      hratio->SetStats(false);   // No statistics on lower plot
      hratio->Divide(h_lastGains.get());
      hratio->SetMarkerStyle(20);
      hratio->Draw("ep");  // Draw the ratio plot

      // Y axis ratio plot settings
      hratio->GetYaxis()->SetTitle(
          ("ratio " + std::to_string(std::get<0>(firstiov)) + " / " + std::to_string(std::get<0>(lastiov))).c_str());

      hratio->GetYaxis()->SetNdivisions(505);

      SiStripPI::makeNicePlotStyle(hratio);

      hratio->GetYaxis()->SetTitleSize(25);
      hratio->GetXaxis()->SetLabelSize(25);

      hratio->GetYaxis()->SetTitleFont(43);
      hratio->GetYaxis()->SetTitleOffset(2.5);
      hratio->GetYaxis()->SetLabelFont(43);  // Absolute font size in pixel (precision 3)
      hratio->GetYaxis()->SetLabelSize(25);

      // X axis ratio plot settings
      hratio->GetXaxis()->SetTitleSize(30);
      hratio->GetXaxis()->SetTitleFont(43);
      hratio->GetXaxis()->SetTitle("SiStrip APV Gains");
      hratio->GetXaxis()->SetLabelFont(43);  // Absolute font size in pixel (precision 3)
      hratio->GetXaxis()->SetTitleOffset(3.);

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  using SiStripApvGainsValuesComparatorSingleTag = SiStripApvGainsValuesComparatorBase<1, MULTI_IOV>;
  using SiStripApvGainsValuesComparatorTwoTags = SiStripApvGainsValuesComparatorBase<2, SINGLE_IOV>;

  //*******************************************//
  // Compare Gains ratio from 2 IOVs, region by region
  //******************************************//

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripApvGainsRatioComparatorByRegionBase : public PlotImage<SiStripApvGain, nIOVs, ntags> {
  public:
    SiStripApvGainsRatioComparatorByRegionBase()
        : PlotImage<SiStripApvGain, nIOVs, ntags>("Module by Module Comparison of SiStrip APV gains"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      //gStyle->SetPalette(5);
      SiStripPI::setPaletteStyle(SiStripPI::GRAY);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripApvGain> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      std::map<std::pair<uint32_t, int>, float> lastmap, firstmap;

      // loop on the last payload
      for (const auto& d : detid) {
        SiStripApvGain::Range range = last_payload->getRange(d);
        float Gain = 0;
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; ++it) {
          nAPV += 1;
          Gain = last_payload->getApvGain(it, range);
          std::pair<uint32_t, int> index = std::make_pair(d, nAPV);
          lastmap[index] = Gain;
        }  // end loop on APVs
      }  // end loop on detids

      detid.clear();
      first_payload->getDetIds(detid);

      // loop on the first payload
      for (const auto& d : detid) {
        SiStripApvGain::Range range = first_payload->getRange(d);
        float Gain = 0;
        float nAPV = 0;
        for (int it = 0; it < range.second - range.first; ++it) {
          nAPV += 1;
          Gain = first_payload->getApvGain(it, range);
          std::pair<uint32_t, int> index = std::make_pair(d, nAPV);
          firstmap[index] = Gain;
        }  // end loop on APVs
      }  // end loop on detids

      TCanvas canvas("Payload comparison by Tracker Region", "payload comparison by Tracker Region", 1800, 800);
      canvas.Divide(2, 1);

      auto h2first = std::make_unique<TH2F>(
          "byRegion1", "SiStrip APV Gain values by region;; average SiStrip Gain", 38, 1., 39., 100., 0., 2.);
      auto h2last = std::make_unique<TH2F>(
          "byRegion2", "SiStrip APV Gain values by region;; average SiStrip Gain", 38, 1., 39., 100., 0., 2.);

      auto h2ratio = std::make_unique<TH2F>("byRegionRatio",
                                            Form("SiStrip APV Gains ratio by region;; Gains ratio IOV: %s/ IOV %s",
                                                 lastIOVsince.c_str(),
                                                 firstIOVsince.c_str()),
                                            38,
                                            1.,
                                            39.,
                                            100.,
                                            0.85,
                                            1.15);

      h2first->SetStats(false);
      h2last->SetStats(false);
      h2ratio->SetStats(false);

      canvas.cd(1)->SetBottomMargin(0.18);
      canvas.cd(1)->SetLeftMargin(0.12);
      canvas.cd(1)->SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      std::string detector;
      std::string currentDetector;

      for (const auto& element : lastmap) {
        auto region = this->getTheRegion(element.first.first);
        auto bin = SiStripPI::regionType(region).first;
        auto label = SiStripPI::regionType(region).second;

        h2last->Fill(bin, element.second);
        h2last->GetXaxis()->SetBinLabel(bin, label);
        h2ratio->Fill(bin, element.second / firstmap[element.first]);
        h2ratio->GetXaxis()->SetBinLabel(bin, label);
      }

      for (const auto& element : firstmap) {
        auto region = this->getTheRegion(element.first.first);
        auto bin = SiStripPI::regionType(region).first;
        auto label = SiStripPI::regionType(region).second;

        h2first->Fill(bin, element.second);
        h2first->GetXaxis()->SetBinLabel(bin, label);
      }

      h2first->GetXaxis()->LabelsOption("v");
      h2last->GetXaxis()->LabelsOption("v");
      h2ratio->GetXaxis()->LabelsOption("v");

      h2last->SetLineColor(kBlue);
      h2first->SetLineColor(kRed);
      h2first->SetFillColor(kRed);

      h2first->SetMarkerStyle(20);
      h2last->SetMarkerStyle(21);

      h2first->SetMarkerColor(kRed);
      h2last->SetMarkerColor(kBlue);

      canvas.cd(1);
      h2first->Draw("BOX");
      h2last->Draw("BOXsame");

      TLegend legend = TLegend(0.70, 0.8, 0.95, 0.9);
      legend.SetHeader("Gain Comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(h2first.get(), ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "F");
      legend.AddEntry(h2last.get(), ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "F");
      legend.Draw("same");

      canvas.cd(2);
      canvas.cd(2)->SetBottomMargin(0.18);
      canvas.cd(2)->SetLeftMargin(0.12);
      canvas.cd(2)->SetRightMargin(0.12);

      h2ratio->Draw("COLZ");
      auto hpfx_tmp = (TProfile*)(h2ratio->ProfileX("_pfx", 1, -1, "o"));
      hpfx_tmp->SetStats(kFALSE);
      hpfx_tmp->SetMarkerColor(kRed);
      hpfx_tmp->SetLineColor(kRed);
      hpfx_tmp->SetMarkerSize(1.2);
      hpfx_tmp->SetMarkerStyle(20);
      hpfx_tmp->Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      delete hpfx_tmp;
      return true;
    }

  private:
    TrackerTopology m_trackerTopo;

    SiStripPI::TrackerRegion getTheRegion(DetId detid) {
      int layer = 0;
      int stereo = 0;
      int detNum = 0;

      switch (detid.subdetId()) {
        case StripSubdetector::TIB:
          layer = m_trackerTopo.tibLayer(detid);
          stereo = m_trackerTopo.tibStereo(detid);
          detNum = 1000;
          break;
        case StripSubdetector::TOB:
          layer = m_trackerTopo.tobLayer(detid);
          stereo = m_trackerTopo.tobStereo(detid);
          detNum = 2000;
          break;
        case StripSubdetector::TEC:
          // is this module in TEC+ or TEC-?
          layer = m_trackerTopo.tecWheel(detid);
          stereo = m_trackerTopo.tecStereo(detid);
          detNum = 3000;
          break;
        case StripSubdetector::TID:
          // is this module in TID+ or TID-?
          layer = m_trackerTopo.tidWheel(detid);
          stereo = m_trackerTopo.tidStereo(detid);
          detNum = 4000;
          break;
      }

      detNum += layer * 10 + stereo;
      return static_cast<SiStripPI::TrackerRegion>(detNum);
    }
  };

  using SiStripApvGainsRatioComparatorByRegionSingleTag = SiStripApvGainsRatioComparatorByRegionBase<1, MULTI_IOV>;
  using SiStripApvGainsRatioComparatorByRegionTwoTags = SiStripApvGainsRatioComparatorByRegionBase<2, SINGLE_IOV>;

  /************************************************
    Compare Gains for each tracker region
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripApvGainsComparatorByRegionBase : public PlotImage<SiStripApvGain, nIOVs, ntags> {
  public:
    SiStripApvGainsComparatorByRegionBase()
        : PlotImage<SiStripApvGain, nIOVs, ntags>("SiStripGains Comparison By Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripApvGain> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripApvGain> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      SiStripDetSummary summaryLastGain{&m_trackerTopo};

      for (const auto& d : detid) {
        SiStripApvGain::Range range = last_payload->getRange(d);
        for (int it = 0; it < range.second - range.first; ++it) {
          summaryLastGain.add(d, last_payload->getApvGain(it, range));
        }
      }

      SiStripDetSummary summaryFirstGain{&m_trackerTopo};

      for (const auto& d : detid) {
        SiStripApvGain::Range range = first_payload->getRange(d);
        for (int it = 0; it < range.second - range.first; ++it) {
          summaryFirstGain.add(d, first_payload->getApvGain(it, range));
        }
      }

      std::map<unsigned int, SiStripDetSummary::Values> firstmap = summaryFirstGain.getCounts();
      std::map<unsigned int, SiStripDetSummary::Values> lastmap = summaryLastGain.getCounts();
      //=========================

      TCanvas canvas("Region summary", "region summary", 1200, 1000);
      canvas.cd();

      auto hfirst = std::make_unique<TH1F>("byRegion1",
                                           "SiStrip APV Gain average by region;; average SiStrip Gain",
                                           firstmap.size(),
                                           0.,
                                           firstmap.size());
      auto hlast = std::make_unique<TH1F>(
          "byRegion2", "SiStrip APV Gain average by region;; average SiStrip Gain", lastmap.size(), 0., lastmap.size());

      hfirst->SetStats(false);
      hlast->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto& element : lastmap) {
        iBin++;
        int count = element.second.count;
        double mean = (element.second.mean) / count;

        if (currentDetector.empty())
          currentDetector = "TIB";

        switch ((element.first) / 1000) {
          case 1:
            detector = "TIB";
            break;
          case 2:
            detector = "TOB";
            break;
          case 3:
            detector = "TEC";
            break;
          case 4:
            detector = "TID";
            break;
        }

        hlast->SetBinContent(iBin, mean);
        hlast->SetBinError(iBin, mean / 10000.);
        hlast->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        hlast->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      // reset the count
      iBin = 0;

      for (const auto& element : firstmap) {
        iBin++;
        int count = element.second.count;
        double mean = (element.second.mean) / count;

        hfirst->SetBinContent(iBin, mean);
        hfirst->SetBinError(iBin, mean / 10000.);
        hfirst->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        hfirst->GetXaxis()->LabelsOption("v");
      }

      auto extrema = SiStripPI::getExtrema(hfirst.get(), hlast.get());
      hlast->GetYaxis()->SetRangeUser(extrema.first, extrema.second);

      hlast->SetMarkerStyle(20);
      hlast->SetMarkerSize(1);
      hlast->Draw("E1");
      hlast->Draw("Psame");

      hfirst->SetMarkerStyle(18);
      hfirst->SetMarkerSize(1);
      hfirst->SetLineColor(kBlue);
      hfirst->SetMarkerColor(kBlue);
      hfirst->Draw("E1same");
      hfirst->Draw("Psame");

      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(
            hfirst->GetBinLowEdge(line), canvas.cd()->GetUymin(), hfirst->GetBinLowEdge(line), canvas.cd()->GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.70, 0.8, 0.95, 0.9);
      legend.SetHeader("Gain Comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(hfirst.get(), ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "PL");
      legend.AddEntry(hlast.get(), ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "PL");
      legend.Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  using SiStripApvGainsComparatorByRegionSingleTag = SiStripApvGainsComparatorByRegionBase<1, MULTI_IOV>;
  using SiStripApvGainsComparatorByRegionTwoTags = SiStripApvGainsComparatorByRegionBase<2, SINGLE_IOV>;

  /************************************************
    Plot gain averages by region 
  *************************************************/

  class SiStripApvGainsByRegion : public PlotImage<SiStripApvGain, SINGLE_IOV> {
  public:
    SiStripApvGainsByRegion()
        : PlotImage<SiStripApvGain, SINGLE_IOV>("SiStripGains By Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload(std::get<1>(iov));

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryGain{&m_trackerTopo};

      for (const auto& d : detid) {
        SiStripApvGain::Range range = payload->getRange(d);
        for (int it = 0; it < range.second - range.first; ++it) {
          summaryGain.add(d, payload->getApvGain(it, range));
        }
      }

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryGain.getCounts();
      //=========================

      TCanvas canvas("Region summary", "region summary", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>(
          "byRegion", "SiStrip Gain average by region;; average SiStrip Gain", map.size(), 0., map.size());
      h1->SetStats(false);
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto& element : map) {
        iBin++;
        int count = element.second.count;
        double mean = (element.second.mean) / count;

        if (currentDetector.empty())
          currentDetector = "TIB";

        switch ((element.first) / 1000) {
          case 1:
            detector = "TIB";
            break;
          case 2:
            detector = "TOB";
            break;
          case 3:
            detector = "TEC";
            break;
          case 4:
            detector = "TID";
            break;
        }

        h1->SetBinContent(iBin, mean);
        h1->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        h1->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("HIST");
      h1->Draw("Psame");

      canvas.Update();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(h1->GetBinLowEdge(line), canvas.GetUymin(), h1->GetBinLowEdge(line), canvas.GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
      legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripApvGain) {
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainCompareByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainRatioByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainDiffByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValuesComparatorSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValuesComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorByRegionSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorByRegionTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioComparatorByRegionSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioComparatorByRegionTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsTHStack);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvBarrelGainsByLayer);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvAbsoluteBarrelGainsByLayer);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvEndcapMinusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvEndcapPlusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvAbsoluteEndcapMinusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvAbsoluteEndcapPlusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAverageTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsDefaultTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaximumTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMinimumTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAvgDeviationRatioWithPreviousIOVTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAvgDeviationRatioTrackerMapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaxDeviationRatioWithPreviousIOVTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaxDeviationRatioTrackerMapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainMin_History);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainMax_History);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainMean_History);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainRMS_History);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIDGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTOBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTECGainByRunMeans);
}
