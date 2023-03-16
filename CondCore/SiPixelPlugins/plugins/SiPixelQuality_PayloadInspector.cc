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
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"

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

  using namespace cond::payloadInspector;

  /************************************************
    test class
  *************************************************/

  class SiPixelQualityTest : public Histogram1D<SiPixelQuality, SINGLE_IOV> {
  public:
    SiPixelQualityTest()
        : Histogram1D<SiPixelQuality, SINGLE_IOV>("SiPixelQuality test", "SiPixelQuality test", 10, 0.0, 10.0) {}

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
    Debugging class, to not be displayed
  *************************************************/

  class SiPixelQualityDebugger : public Histogram1D<SiPixelQuality, SINGLE_IOV> {
  public:
    SiPixelQualityDebugger()
        : Histogram1D<SiPixelQuality, SINGLE_IOV>("SiPixelQuality test", "SiPixelQuality test", 10, 0.0, 10.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiPixelQuality> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          SiPixelPI::topolInfo t_info_fromXML;
          t_info_fromXML.init();

          auto theDisabledModules = payload->getBadComponentList();
          for (const auto& mod : theDisabledModules) {
            DetId detid(mod.DetID);
            auto PhInfo = SiPixelPI::PhaseInfo(SiPixelPI::phase1size);
            const char* path_toTopologyXML = PhInfo.pathToTopoXML();
            auto tTopo =
                StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());
            t_info_fromXML.fillGeometryInfo(detid, tTopo, PhInfo.phase());
            std::stringstream ss;
            t_info_fromXML.printAll(ss);
            std::bitset<16> bad_rocs(mod.BadRocs);

            if (t_info_fromXML.subDetId() == 1 && t_info_fromXML.layer() == 1) {
              std::cout << ss.str() << " s_module: " << SiPixelPI::signed_module(mod.DetID, tTopo, true)
                        << " s_ladder: " << SiPixelPI::signed_ladder(mod.DetID, tTopo, true)
                        << " error type:" << mod.errorType << " BadRocs: " << bad_rocs.to_string('O', 'X') << std::endl;
            }
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
    summary class
  *************************************************/

  class SiPixelQualityBadRocsSummary : public PlotImage<SiPixelQuality, MULTI_IOV, 1> {
  public:
    SiPixelQualityBadRocsSummary() : PlotImage<SiPixelQuality, MULTI_IOV, 1>("SiPixel Quality Summary") {}

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
      TCanvas canv("Partition summary", "partition summary", 1200, 1000);
      canv.SetBottomMargin(0.11);
      canv.SetLeftMargin(0.13);
      canv.SetRightMargin(0.05);
      canv.cd();
      SiPixelPI::displayNotSupported(canv, 0);

      std::string fileName(m_imageFileName);
      canv.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
    time history class
  *************************************************/

  class SiPixelQualityBadRocsTimeHistory : public TimeHistoryPlot<SiPixelQuality, std::pair<double, double> > {
  public:
    SiPixelQualityBadRocsTimeHistory()
        : TimeHistoryPlot<SiPixelQuality, std::pair<double, double> >("bad ROCs count vs time", "bad ROCs count") {}

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
  class SiPixelQualityMap : public PlotImage<SiPixelQuality, SINGLE_IOV> {
  public:
    SiPixelQualityMap()
        : PlotImage<SiPixelQuality, SINGLE_IOV>("SiPixelQuality Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));

      Phase1PixelROCMaps theMap("");

      auto theDisabledModules = payload->getBadComponentList();
      if (this->isPhase0(theDisabledModules)) {
        edm::LogError("SiPixelQuality_PayloadInspector")
            << "SiPixelQuality maps are not supported for non-Phase1 Pixel geometries !";
        TCanvas canvas("Canv", "Canv", 1200, 1000);
        SiPixelPI::displayNotSupported(canvas, 0);
        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

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

    //_________________________________________________
    bool isPhase0(std::vector<SiPixelQuality::disabledModuleType> mods) {
      SiPixelDetInfoFileReader reader =
          SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh0DefaultFile).fullPath());
      const auto& p0detIds = reader.getAllDetIds();

      std::vector<uint32_t> ownDetIds;
      std::transform(mods.begin(),
                     mods.end(),
                     std::back_inserter(ownDetIds),
                     [](SiPixelQuality::disabledModuleType d) -> uint32_t { return d.DetID; });

      for (const auto& det : ownDetIds) {
        // if found at least one phase-0 detId early return
        if (std::find(p0detIds.begin(), p0detIds.end(), det) != p0detIds.end()) {
          return true;
        }
      }
      return false;
    }
  };

  using SiPixelBPixQualityMap = SiPixelQualityMap<SiPixelPI::t_barrel>;
  using SiPixelFPixQualityMap = SiPixelQualityMap<SiPixelPI::t_forward>;
  using SiPixelFullQualityMap = SiPixelQualityMap<SiPixelPI::t_all>;

  /************************************************
   occupancy style map whole Pixel, difference of payloads
  *************************************************/
  template <SiPixelPI::DetType myType, IOVMultiplicity nIOVs, int ntags>
  class SiPixelQualityMapComparisonBase : public PlotImage<SiPixelQuality, nIOVs, ntags> {
  public:
    SiPixelQualityMapComparisonBase()
        : PlotImage<SiPixelQuality, nIOVs, ntags>(
              Form("SiPixelQuality %s Pixel Map", SiPixelPI::DetNames[myType].c_str())),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto f_tagname = PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        l_tagname = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiPixelQuality> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiPixelQuality> first_payload = this->fetchPayload(std::get<1>(firstiov));

      if (this->isPhase0(first_payload) || this->isPhase0(last_payload)) {
        edm::LogError("SiPixelQuality_PayloadInspector")
            << "SiPixelQuality comparison maps are not supported for non-Phase1 Pixel geometries !";
        TCanvas canvas("Canv", "Canv", 1200, 1000);
        SiPixelPI::displayNotSupported(canvas, 0);
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      Phase1PixelROCMaps theMap("", "#Delta payload A - payload B");

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

      // first loop on the first payload (newest)
      fillTheMapFromPayload(theMap, first_payload, false);

      // then loop on the second payload (oldest)
      fillTheMapFromPayload(theMap, last_payload, true);  // true will subtract

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

    //_________________________________________________
    bool isPhase0(const std::shared_ptr<SiPixelQuality>& payload) {
      const auto mods = payload->getBadComponentList();
      SiPixelDetInfoFileReader reader =
          SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh0DefaultFile).fullPath());
      const auto& p0detIds = reader.getAllDetIds();

      std::vector<uint32_t> ownDetIds;
      std::transform(mods.begin(),
                     mods.end(),
                     std::back_inserter(ownDetIds),
                     [](SiPixelQuality::disabledModuleType d) -> uint32_t { return d.DetID; });

      for (const auto& det : ownDetIds) {
        // if found at least one phase-0 detId early return
        if (std::find(p0detIds.begin(), p0detIds.end(), det) != p0detIds.end()) {
          return true;
        }
      }
      return false;
    }

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

  using SiPixelBPixQualityMapCompareSingleTag = SiPixelQualityMapComparisonBase<SiPixelPI::t_barrel, MULTI_IOV, 1>;
  using SiPixelFPixQualityMapCompareSingleTag = SiPixelQualityMapComparisonBase<SiPixelPI::t_forward, MULTI_IOV, 1>;
  using SiPixelFullQualityMapCompareSingleTag = SiPixelQualityMapComparisonBase<SiPixelPI::t_all, MULTI_IOV, 1>;
  using SiPixelBPixQualityMapCompareTwoTags = SiPixelQualityMapComparisonBase<SiPixelPI::t_barrel, SINGLE_IOV, 2>;
  using SiPixelFPixQualityMapCompareTwoTags = SiPixelQualityMapComparisonBase<SiPixelPI::t_forward, SINGLE_IOV, 2>;
  using SiPixelFullQualityMapCompareTwoTags = SiPixelQualityMapComparisonBase<SiPixelPI::t_all, SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQuality) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsSummary);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsTimeHistory);
  //PAYLOAD_INSPECTOR_CLASS(SiPixelQualityDebugger);
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
