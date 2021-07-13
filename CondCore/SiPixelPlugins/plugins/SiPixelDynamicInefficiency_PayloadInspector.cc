/*!
  \file SiPixelDynamicInefficiency_PayloadInspector
  \Payload Inspector Plugin for SiPixelDynamicInefficiency
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
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
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
  namespace SiPixDynIneff {

    // constants for ROC level simulation for Phase1
    enum shiftEnumerator { FPixRocIdShift = 3, BPixRocIdShift = 6 };
    static const int rocIdMaskBits = 0x1F;

    struct packedBadRocFraction {
      std::vector<int> badRocNumber;
      std::vector<float> badRocFrac;
    };

    using BRFractions = std::unordered_map<uint32_t, packedBadRocFraction>;

    //_________________________________________________
    BRFractions pbrf(std::shared_ptr<SiPixelDynamicInefficiency> payload) {
      BRFractions f;
      const std::map<uint32_t, double>& PixelGeomFactorsDBIn = payload->getPixelGeomFactors();

      // first fill
      for (const auto db_factor : PixelGeomFactorsDBIn) {
        int subid = DetId(db_factor.first).subdetId();
        int shift = (subid == static_cast<int>(PixelSubdetector::PixelBarrel)) ? BPixRocIdShift : FPixRocIdShift;
        unsigned int rocMask = rocIdMaskBits << shift;
        unsigned int rocId = (((db_factor.first) & rocMask) >> shift);
        uint32_t rawid = db_factor.first & (~rocMask);

        if (f.find(rawid) == f.end()) {
          packedBadRocFraction p;
          f.insert(std::make_pair(rawid, p));
        }

        if (rocId != 0) {
          rocId--;
          double factor = db_factor.second;
          double badFraction = 1 - factor;

          f.at(rawid).badRocNumber.emplace_back(rocId);
          f.at(rawid).badRocFrac.emplace_back(badFraction);
        }
      }
      return f;
    }

    //_________________________________________________
    bool isPhase0(const BRFractions& fractions) {
      SiPixelDetInfoFileReader reader =
          SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh0DefaultFile).fullPath());
      const auto& p0detIds = reader.getAllDetIds();
      std::vector<uint32_t> ownDetIds;

      std::transform(fractions.begin(),
                     fractions.end(),
                     std::back_inserter(ownDetIds),
                     [](std::pair<uint32_t, packedBadRocFraction> d) -> uint32_t { return d.first; });

      for (const auto& det : ownDetIds) {
        // if found at least one phase-0 detId early return
        if (std::find(p0detIds.begin(), p0detIds.end(), det) != p0detIds.end()) {
          return true;
        }
      }
      return false;
    }
  }  // namespace SiPixDynIneff

  /************************************************
    test class
  *************************************************/

  class SiPixelDynamicInefficiencyTest : public Histogram1D<SiPixelDynamicInefficiency, SINGLE_IOV> {
  public:
    SiPixelDynamicInefficiencyTest()
        : Histogram1D<SiPixelDynamicInefficiency, SINGLE_IOV>(
              "SiPixelDynamicInefficiency test", "SiPixelDynamicInefficiency test", 1, 0.0, 1.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiPixelDynamicInefficiency> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          const auto geomFactors = payload->getPixelGeomFactors();
          for (const auto [ID, value] : geomFactors) {
            std::cout << ID << " : " << value << std::endl;
            ;
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
   occupancy style map whole Pixel of inefficient ROCs
  *************************************************/
  template <SiPixelPI::DetType myType>
  class SiPixelIneffROCfromDynIneffMap : public PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV> {
  public:
    SiPixelIneffROCfromDynIneffMap()
        : PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV>("SiPixel Inefficient ROC from Dyn Ineff Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiPixelDynamicInefficiency> payload = fetchPayload(std::get<1>(iov));

      const auto fr = SiPixDynIneff::pbrf(payload);

      if (SiPixDynIneff::isPhase0(fr)) {
        edm::LogError("SiPixelDynamicInefficiency_PayloadInspector")
            << "SiPixelIneffROCfromDynIneff maps are not supported for non-Phase1 Pixel geometries !";
        TCanvas canvas("Canv", "Canv", 1200, 1000);
        SiPixelPI::displayNotSupported(canvas, 0);
        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      Phase1PixelROCMaps theMap("", "bad pixel fraction in ROC [%]");

      for (const auto& element : fr) {
        auto rawid = element.first;
        int subid = DetId(rawid).subdetId();
        auto packedinfo = element.second;
        auto badRocs = packedinfo.badRocNumber;
        auto badRocsF = packedinfo.badRocFrac;

        for (size_t i = 0; i < badRocs.size(); i++) {
          std::bitset<16> rocToMark;
          rocToMark.set(badRocs[i]);
          if ((subid == PixelSubdetector::PixelBarrel && myType == SiPixelPI::t_barrel) ||
              (subid == PixelSubdetector::PixelEndcap && myType == SiPixelPI::t_forward) ||
              (myType == SiPixelPI::t_all)) {
            theMap.fillSelectedRocs(rawid, rocToMark, badRocsF[i] * 100.f);
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
          throw cms::Exception("SiPixelIneffROCfromDynIneffMap")
              << "\nERROR: unrecognized Pixel Detector part " << std::endl;
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

  using SiPixelBPixIneffROCfromDynIneffMap = SiPixelIneffROCfromDynIneffMap<SiPixelPI::t_barrel>;
  using SiPixelFPixIneffROCfromDynIneffMap = SiPixelIneffROCfromDynIneffMap<SiPixelPI::t_forward>;
  using SiPixelFullIneffROCfromDynIneffMap = SiPixelIneffROCfromDynIneffMap<SiPixelPI::t_all>;

  /************************************************
   occupancy style map whole Pixel, difference of payloads
  *************************************************/
  template <SiPixelPI::DetType myType, IOVMultiplicity nIOVs, int ntags>
  class SiPixelIneffROCComparisonBase : public PlotImage<SiPixelDynamicInefficiency, nIOVs, ntags> {
  public:
    SiPixelIneffROCComparisonBase()
        : PlotImage<SiPixelDynamicInefficiency, nIOVs, ntags>(
              Form("SiPixelDynamicInefficiency %s Pixel Map", SiPixelPI::DetNames[myType].c_str())),
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

      std::shared_ptr<SiPixelDynamicInefficiency> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiPixelDynamicInefficiency> first_payload = this->fetchPayload(std::get<1>(firstiov));

      const auto fp = SiPixDynIneff::pbrf(last_payload);
      const auto lp = SiPixDynIneff::pbrf(first_payload);

      if (SiPixDynIneff::isPhase0(fp) || SiPixDynIneff::isPhase0(lp)) {
        edm::LogError("SiPixelDynamicInefficiency_PayloadInspector")
            << "SiPixelDynamicInefficiency comparison maps are not supported for non-Phase1 Pixel geometries !";
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
        headerText =
            fmt::sprintf("#color[2]{A: %s, %s} - #color[4]{B: %s, %s}", f_tagname, f_IOVstring, l_tagname, l_IOVstring);
      } else {
        headerText = fmt::sprintf("%s,IOV #color[2]{A: %s} - #color[4]{B: %s} ", f_tagname, f_IOVstring, l_IOVstring);
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
          throw cms::Exception("SiPixelDynamicInefficiencyMapComparison")
              << "\nERROR: unrecognized Pixel Detector part " << std::endl;
      }

      // first loop on the first payload (newest)
      fillTheMapFromPayload(theMap, fp, false);

      // then loop on the second payload (oldest)
      fillTheMapFromPayload(theMap, lp, true);  // true will subtract

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
    void fillTheMapFromPayload(Phase1PixelROCMaps& theMap, const SiPixDynIneff::BRFractions& fr, bool subtract) {
      for (const auto& element : fr) {
        auto rawid = element.first;
        int subid = DetId(rawid).subdetId();
        auto packedinfo = element.second;
        auto badRocs = packedinfo.badRocNumber;
        auto badRocsF = packedinfo.badRocFrac;

        for (size_t i = 0; i < badRocs.size(); i++) {
          std::bitset<16> rocToMark;
          rocToMark.set(badRocs[i]);
          if ((subid == PixelSubdetector::PixelBarrel && myType == SiPixelPI::t_barrel) ||
              (subid == PixelSubdetector::PixelEndcap && myType == SiPixelPI::t_forward) ||
              (myType == SiPixelPI::t_all)) {
            theMap.fillSelectedRocs(rawid, rocToMark, badRocsF[i] * (subtract ? -1. : 1.));
          }
        }
      }
    }
  };

  /*
  using SiPixelBPixIneffROCsMapCompareSingleTag = SiPixelIneffROCComparisonBase<SiPixelPI::t_barrel, MULTI_IOV, 1>;
  using SiPixelFPixIneffROCsMapCompareSingleTag = SiPixelIneffROCComparisonBase<SiPixelPI::t_forward, MULTI_IOV, 1>;
  using SiPixelFullIneffROCsMapCompareSingleTag = SiPixelIneffROCComparisonBase<SiPixelPI::t_all, MULTI_IOV, 1>;
  */
  using SiPixelBPixIneffROCsMapCompareTwoTags = SiPixelIneffROCComparisonBase<SiPixelPI::t_barrel, SINGLE_IOV, 2>;
  using SiPixelFPixIneffROCsMapCompareTwoTags = SiPixelIneffROCComparisonBase<SiPixelPI::t_forward, SINGLE_IOV, 2>;
  using SiPixelFullIneffROCsMapCompareTwoTags = SiPixelIneffROCComparisonBase<SiPixelPI::t_all, SINGLE_IOV, 2>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelDynamicInefficiency) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelDynamicInefficiencyTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixIneffROCfromDynIneffMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixIneffROCfromDynIneffMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullIneffROCfromDynIneffMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixIneffROCsMapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixIneffROCsMapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullIneffROCsMapCompareTwoTags);
}
