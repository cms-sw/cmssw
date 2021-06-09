/*!
  \file SiPixelQuality_PayloadInspector
  \Payload Inspector Plugin for SiPixelQuality
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/10/18 14:48:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  //using namespace cond::payloadInspector;

  /************************************************
    test class
  *************************************************/

  class SiPixelQualityTest
      : public cond::payloadInspector::Histogram1D<SiPixelQuality, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelQualityTest()
        : cond::payloadInspector::Histogram1D<SiPixelQuality, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelQuality test", "SiPixelQuality test", 10, 0.0, 10.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiPixelQuality> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          auto theDisabledModules = payload->getBadComponentList();
          for (const auto& mod : theDisabledModules) {
            int BadRocCount(0);
            for (unsigned short n = 0; n < 16; n++) {
              unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
              if (mod.BadRocs & mask)
                BadRocCount++;
            }
            COUT << "detId:" << mod.DetID << " error type:" << mod.errorType << " BadRocs:" << BadRocCount << std::endl;
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
    summary class
  *************************************************/

  class SiPixelQualityBadRocsSummary
      : public cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::MULTI_IOV, 1> {
  public:
    SiPixelQualityBadRocsSummary()
        : cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::MULTI_IOV, 1>(
              "SiPixel Quality Summary") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (const auto& iov : tag.iovs) {
        std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));
        auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

        COUT << "======================= " << unpacked.first << " : " << unpacked.second << std::endl;
        auto theDisabledModules = payload->getBadComponentList();
        for (const auto& mod : theDisabledModules) {
          COUT << "detId: " << mod.DetID << " |error type: " << mod.errorType << " |BadRocs: " << mod.BadRocs
               << std::endl;
        }
      }

      //=========================
      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      SiPixelPI::displayNotSupported(canvas, 0);
      canvas.cd();
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
    time history class
  *************************************************/

  class SiPixelQualityBadRocsTimeHistory
      : public cond::payloadInspector::TimeHistoryPlot<SiPixelQuality, std::pair<double, double> > {
  public:
    SiPixelQualityBadRocsTimeHistory()
        : cond::payloadInspector::TimeHistoryPlot<SiPixelQuality, std::pair<double, double> >("bad ROCs count vs time",
                                                                                              "bad ROCs count") {}

    std::pair<double, double> getFromPayload(SiPixelQuality& payload) override {
      return std::make_pair(extractBadRocCount(payload), 0.);
    }

    unsigned int extractBadRocCount(SiPixelQuality& payload) {
      unsigned int BadRocCount(0);
      auto theDisabledModules = payload.getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        for (unsigned short n = 0; n < 16; n++) {
          unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
          if (mod.BadRocs & mask)
            BadRocCount++;
        }
      }
      return BadRocCount;
    }
  };

  /************************************************
   occupancy style map whole Pixel
  *************************************************/
  template <SiPixelPI::DetType myType>
  class SiPixelQualityMap
      : public cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelQualityMap()
        : cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelQuality Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));

      Phase1PixelROCMaps theMap("");

      auto theDisabledModules = payload->getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        int subid = DetId(mod.DetID).subdetId();

        if ((subid == PixelSubdetector::PixelBarrel && myType == SiPixelPI::t_barrel) ||
            (subid == PixelSubdetector::PixelEndcap && myType == SiPixelPI::t_forward) ||
            (myType == SiPixelPI::t_all)) {
          std::bitset<16> bad_rocs(mod.BadRocs);
          if (payload->IsModuleBad(mod.DetID)) {
            theMap.fillWholeModule(mod.DetID, 1.);
          } else {
            theMap.fillSelectedRocs(mod.DetID, bad_rocs, 1.);
          }
        }
      }

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, k_height[myType]);
      canvas.cd();

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      std::string IOVstring = (unpacked.first == 0)
                                  ? std::to_string(unpacked.second)
                                  : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second));

      const auto headerText = fmt::sprintf("#color[4]{%s},  IOV: #color[4]{%s}", tagname, IOVstring);

      switch (myType) {
        case SiPixelPI::t_barrel:
          theMap.drawBarrelMaps(canvas, headerText);
          break;
        case SiPixelPI::t_forward:
          theMap.drawForwardMaps(canvas, headerText);
          break;
        case SiPixelPI::t_all:
          theMap.drawMaps(canvas, headerText);
          break;
        default:
          throw cms::Exception("SiPixelQualityMap") << "\nERROR: unrecognized Pixel Detector part " << std::endl;
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outAll.root");
#endif

      return true;
    }

  private:
    static constexpr std::array<int, 3> k_height = {{1200, 600, 1600}};
    TrackerTopology m_trackerTopo;
  };

  using SiPixelBPixQualityMap = SiPixelQualityMap<SiPixelPI::t_barrel>;
  using SiPixelFPixQualityMap = SiPixelQualityMap<SiPixelPI::t_forward>;
  using SiPixelFullQualityMap = SiPixelQualityMap<SiPixelPI::t_all>;

  /************************************************
   occupancy style map whole Pixel, difference of payloads
  *************************************************/
  template <SiPixelPI::DetType myType, cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class SiPixelQualityMapComparisonBase : public cond::payloadInspector::PlotImage<SiPixelQuality, nIOVs, ntags> {
  public:
    SiPixelQualityMapComparisonBase()
        : cond::payloadInspector::PlotImage<SiPixelQuality, nIOVs, ntags>(
              Form("SiPixelQuality %s Pixel Map", SiPixelPI::DetNames[myType].c_str())),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiPixelQuality> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiPixelQuality> first_payload = this->fetchPayload(std::get<1>(firstiov));

      Phase1PixelROCMaps theMap("", "#Delta payload A - payload B");

      // first loop on the first payload (newest)
      fillTheMapFromPayload(theMap, first_payload, false);

      // then loop on the second payload (oldest)
      fillTheMapFromPayload(theMap, last_payload, true);  // true will subtract

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, k_height[myType]);
      canvas.cd();

      auto f_unpacked = SiPixelPI::unpack(std::get<0>(firstiov));
      auto l_unpacked = SiPixelPI::unpack(std::get<0>(lastiov));

      std::string f_IOVstring = (f_unpacked.first == 0)
                                    ? std::to_string(f_unpacked.second)
                                    : (std::to_string(f_unpacked.first) + "," + std::to_string(f_unpacked.second));

      std::string l_IOVstring = (l_unpacked.first == 0)
                                    ? std::to_string(l_unpacked.second)
                                    : (std::to_string(l_unpacked.first) + "," + std::to_string(l_unpacked.second));

      std::string headerText;

      if (this->m_plotAnnotations.ntags == 2) {
        headerText = fmt::sprintf(
            "#Delta #color[2]{A: %s, %s} - #color[4]{B: %s, %s}", f_tagname, f_IOVstring, l_tagname, l_IOVstring);
      } else {
        headerText =
            fmt::sprintf("%s, #Delta IOV #color[2]{A: %s} - #color[4]{B: %s} ", f_tagname, f_IOVstring, l_IOVstring);
      }

      switch (myType) {
        case SiPixelPI::t_barrel:
          theMap.drawBarrelMaps(canvas, headerText);
          break;
        case SiPixelPI::t_forward:
          theMap.drawForwardMaps(canvas, headerText);
          break;
        case SiPixelPI::t_all:
          theMap.drawMaps(canvas, headerText);
          break;
        default:
          throw cms::Exception("SiPixelQualityMapComparison")
              << "\nERROR: unrecognized Pixel Detector part " << std::endl;
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outAll.root");
#endif

      return true;
    }

  private:
    static constexpr std::array<int, 3> k_height = {{1200, 600, 1600}};
    TrackerTopology m_trackerTopo;

    //____________________________________________________________________________________________
    void fillTheMapFromPayload(Phase1PixelROCMaps& theMap,
                               const std::shared_ptr<SiPixelQuality>& payload,
                               bool subtract) {
      const auto theDisabledModules = payload->getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        int subid = DetId(mod.DetID).subdetId();
        if ((subid == PixelSubdetector::PixelBarrel && myType == SiPixelPI::t_barrel) ||
            (subid == PixelSubdetector::PixelEndcap && myType == SiPixelPI::t_forward) ||
            (myType == SiPixelPI::t_all)) {
          std::bitset<16> bad_rocs(mod.BadRocs);
          if (payload->IsModuleBad(mod.DetID)) {
            theMap.fillWholeModule(mod.DetID, (subtract ? -1. : 1.));
          } else {
            theMap.fillSelectedRocs(mod.DetID, bad_rocs, (subtract ? -1. : 1.));
          }
        }
      }
    }
  };

  using SiPixelBPixQualityMapCompareSingleTag =
      SiPixelQualityMapComparisonBase<SiPixelPI::t_barrel, cond::payloadInspector::MULTI_IOV, 1>;
  using SiPixelFPixQualityMapCompareSingleTag =
      SiPixelQualityMapComparisonBase<SiPixelPI::t_forward, cond::payloadInspector::MULTI_IOV, 1>;
  using SiPixelFullQualityMapCompareSingleTag =
      SiPixelQualityMapComparisonBase<SiPixelPI::t_all, cond::payloadInspector::MULTI_IOV, 1>;
  using SiPixelBPixQualityMapCompareTwoTags =
      SiPixelQualityMapComparisonBase<SiPixelPI::t_barrel, cond::payloadInspector::MULTI_IOV, 2>;
  using SiPixelFPixQualityMapCompareTwoTags =
      SiPixelQualityMapComparisonBase<SiPixelPI::t_forward, cond::payloadInspector::MULTI_IOV, 2>;
  using SiPixelFullQualityMapCompareTwoTags =
      SiPixelQualityMapComparisonBase<SiPixelPI::t_all, cond::payloadInspector::MULTI_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQuality) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsSummary);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsTimeHistory);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixQualityMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixQualityMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullQualityMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixQualityMapCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixQualityMapCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullQualityMapCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixQualityMapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixQualityMapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullQualityMapCompareTwoTags);
}
