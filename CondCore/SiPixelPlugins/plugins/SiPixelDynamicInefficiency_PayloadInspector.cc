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
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"

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

    // different types of geometrical inefficiency factors
    enum factor { geom = 0, colgeom = 1, chipgeom = 2, pu = 3, INVALID = 4 };
    const std::array<std::string, 5> factorString = {
        {"pixel geometry", "column geometry", "chip geometry", "PU", "invalid"}};

    using FactorMap = std::map<unsigned int, double>;

    // constants for ROC level simulation for Phase1
    enum shiftEnumerator { FPixRocIdShift = 3, BPixRocIdShift = 6 };
    const int rocIdMaskBits = 0x1F;

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

    //_________________________________________________
    double getMatchingGeomFactor(const DetId& detid,
                                 const std::map<unsigned int, double>& map_geomfactor,
                                 const std::vector<uint32_t>& detIdmasks) {
      double geomfactor_db = 1;
      for (auto map_element : map_geomfactor) {
        const DetId mapid = DetId(map_element.first);
        if (mapid.subdetId() != detid.subdetId())
          continue;
        size_t __i = 0;
        for (; __i < detIdmasks.size(); __i++) {
          DetId maskid = DetId(detIdmasks.at(__i));
          if (maskid.subdetId() != mapid.subdetId())
            continue;
          if ((detid.rawId() & maskid.rawId()) != (mapid.rawId() & maskid.rawId()) &&
              (mapid.rawId() & maskid.rawId()) != DetId(mapid.det(), mapid.subdetId()).rawId())
            break;
        }
        if (__i != detIdmasks.size())
          continue;
        geomfactor_db *= map_element.second;
      }
      return geomfactor_db;
    }

    //_________________________________________________
    std::vector<double> getMatchingPUFactors(const DetId& detid,
                                             const std::map<unsigned int, std::vector<double> >& map_pufactory,
                                             const std::vector<uint32_t>& detIdmasks) {
      std::vector<double> pufactors_db;
      for (auto map_element : map_pufactory) {
        const DetId mapid = DetId(map_element.first);
        if (mapid.subdetId() != detid.subdetId())
          continue;
        size_t __i = 0;
        for (; __i < detIdmasks.size(); __i++) {
          DetId maskid = DetId(detIdmasks.at(__i));
          if (maskid.subdetId() != mapid.subdetId())
            continue;
          if ((detid.rawId() & maskid.rawId()) != (mapid.rawId() & maskid.rawId()) &&
              (mapid.rawId() & maskid.rawId()) != DetId(mapid.det(), mapid.subdetId()).rawId())
            break;
        }
        if (__i != detIdmasks.size())
          continue;
        pufactors_db = map_element.second;
      }
      return pufactors_db;
    }

    //_________________________________________________
    bool matches(const DetId& detid, const DetId& db_id, const std::vector<uint32_t>& DetIdmasks) {
      if (detid.subdetId() != db_id.subdetId())
        return false;
      for (size_t i = 0; i < DetIdmasks.size(); ++i) {
        DetId maskid = DetId(DetIdmasks.at(i));
        if (maskid.subdetId() != db_id.subdetId())
          continue;
        if ((detid.rawId() & maskid.rawId()) != (db_id.rawId() & maskid.rawId()) &&
            (db_id.rawId() & maskid.rawId()) != DetId(db_id.det(), db_id.subdetId()).rawId())
          return false;
      }
      return true;
    }

    //_________________________________________________
    bool isPhase1(const std::map<unsigned int, double>& map_geomfactor, const std::vector<uint32_t>& detIdmasks) {
      SiPixelDetInfoFileReader reader =
          SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh1DefaultFile).fullPath());
      const auto& p1detIds = reader.getAllDetIds();

      for (const auto& det : p1detIds) {
        const DetId detid = DetId(det);
        for (const auto& element : map_geomfactor) {
          if (SiPixDynIneff::matches(detid, DetId(element.first), detIdmasks)) {
            return true;
          }
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

          std::map<unsigned int, double> map_pixelgeomfactor = payload->getPixelGeomFactors();
          std::map<unsigned int, double> map_colgeomfactor = payload->getColGeomFactors();
          std::map<unsigned int, double> map_chipgeomfactor = payload->getChipGeomFactors();
          std::map<unsigned int, std::vector<double> > map_pufactor = payload->getPUFactors();
          std::vector<uint32_t> detIdmasks_db = payload->getDetIdmasks();
          double theInstLumiScaleFactor_db = payload->gettheInstLumiScaleFactor_();

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "-------------------------------------------------------";
          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "Printing out DB content:\n";

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "  PixelGeomFactors:";
          for (auto pixel : map_pixelgeomfactor)
            edm::LogPrint("SiPixelDynamicInefficiencyTest")
                << "    MapID = " << pixel.first << "\tFactor = " << pixel.second;
          edm::LogPrint("SiPixelDynamicInefficiencyTest");

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "  ColGeomFactors:";
          for (auto col : map_colgeomfactor)
            edm::LogPrint("SiPixelDynamicInefficiencyTest")
                << "    MapID = " << col.first << "\tFactor = " << col.second;
          edm::LogPrint("SiPixelDynamicInefficiencyTest");

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "  ChipGeomFactors:";
          for (auto chip : map_chipgeomfactor)
            edm::LogPrint("SiPixelDynamicInefficiencyTest")
                << "    MapID = " << chip.first << "\tFactor = " << chip.second;
          edm::LogPrint("SiPixelDynamicInefficiencyTest");

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "  PUFactors:";
          for (auto pu : map_pufactor) {
            edm::LogPrint("SiPixelDynamicInefficiencyTest")
                << "    MapID = " << pu.first << "\t Factor" << (pu.second.size() > 1 ? "s" : "") << " = ";
            for (size_t i = 0, n = pu.second.size(); i < n; ++i)
              edm::LogPrint("SiPixelDynamicInefficiencyTest") << pu.second[i] << ((i == n - 1) ? "\n" : ", ");
          }
          edm::LogPrint("SiPixelDynamicInefficiencyTest");

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "  DetIdmasks:";
          for (auto mask : detIdmasks_db)
            edm::LogPrint("SiPixelDynamicInefficiencyTest") << "    MaskID = " << mask;
          edm::LogPrint("SiPixelDynamicInefficiencyTest");

          edm::LogPrint("SiPixelDynamicInefficiencyTest") << "  theInstLumiScaleFactor = " << theInstLumiScaleFactor_db;

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
          throw cms::Exception("SiPixelIneffROCfromDynIneffMap") << "\nERROR: unrecognized Pixel Detector part ";
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
              << "\nERROR: unrecognized Pixel Detector part ";
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
    These are not implemented for the time being, since the SiPixelDynamicInefficiency is a condition
    used only in simulation, hence there is no such thing as a multi-IoV Dynamic Inefficiency tag 
  */

  using SiPixelBPixIneffROCsMapCompareSingleTag = SiPixelIneffROCComparisonBase<SiPixelPI::t_barrel, MULTI_IOV, 1>;
  using SiPixelFPixIneffROCsMapCompareSingleTag = SiPixelIneffROCComparisonBase<SiPixelPI::t_forward, MULTI_IOV, 1>;
  using SiPixelFullIneffROCsMapCompareSingleTag = SiPixelIneffROCComparisonBase<SiPixelPI::t_all, MULTI_IOV, 1>;

  using SiPixelBPixIneffROCsMapCompareTwoTags = SiPixelIneffROCComparisonBase<SiPixelPI::t_barrel, SINGLE_IOV, 2>;
  using SiPixelFPixIneffROCsMapCompareTwoTags = SiPixelIneffROCComparisonBase<SiPixelPI::t_forward, SINGLE_IOV, 2>;
  using SiPixelFullIneffROCsMapCompareTwoTags = SiPixelIneffROCComparisonBase<SiPixelPI::t_all, SINGLE_IOV, 2>;

  /************************************************
   Full Pixel Tracker Map class
  *************************************************/
  template <SiPixDynIneff::factor theFactor>
  class SiPixelDynamicInefficiencyFullPixelMap : public PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV> {
  public:
    SiPixelDynamicInefficiencyFullPixelMap()
        : PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV>("SiPixelDynamicInefficiency Map") {
      label_ = "SiPixelDynamicInefficiencyFullPixelMap";
      payloadString = fmt::sprintf("%s Dynamic Inefficiency", SiPixDynIneff::factorString[theFactor]);
    }

    bool fill() override {
      gStyle->SetPalette(1);
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelDynamicInefficiency> payload = this->fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        Phase1PixelSummaryMap fullMap("", fmt::sprintf("%s", payloadString), fmt::sprintf("%s", payloadString));
        fullMap.createTrackerBaseMap();

        SiPixDynIneff::FactorMap theMap{};
        switch (theFactor) {
          case SiPixDynIneff::geom:
            theMap = payload->getPixelGeomFactors();
            break;
          case SiPixDynIneff::colgeom:
            theMap = payload->getColGeomFactors();
            break;
          case SiPixDynIneff::chipgeom:
            theMap = payload->getChipGeomFactors();
            break;
          default:
            throw cms::Exception(label_) << "\nERROR: unrecognized type of geometry factor ";
        }

        std::vector<uint32_t> detIdmasks_db = payload->getDetIdmasks();

        if (!SiPixDynIneff::isPhase1(theMap, detIdmasks_db)) {
          edm::LogError(label_) << label_ << " maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, 0);
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        }

        SiPixelDetInfoFileReader reader =
            SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh1DefaultFile).fullPath());
        const auto& p1detIds = reader.getAllDetIds();
        for (const auto& det : p1detIds) {
          const auto& value = SiPixDynIneff::getMatchingGeomFactor(det, theMap, detIdmasks_db);
          fullMap.fillTrackerMap(det, value);
        }

        const auto& range = fullMap.getZAxisRange();
        if (range.first == range.second) {
          // in case the map is completely filled with one value;
          // set the z-axis to be meaningful
          fullMap.setZAxisRange(range.first - 0.01, range.second + 0.01);
        }

        TCanvas canvas("Canv", "Canv", 3000, 2000);
        fullMap.printTrackerMap(canvas);

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.025);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin() + 0.01,
            gPad->GetBottomMargin() + 0.01,
            ("#color[4]{" + tag.name + "}, IOV: #color[4]{" + std::to_string(std::get<0>(iov)) + "}").c_str());

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

  protected:
    std::string payloadString;
    std::string label_;
  };

  using SiPixelDynamicInefficiencyGeomFactorMap = SiPixelDynamicInefficiencyFullPixelMap<SiPixDynIneff::geom>;
  using SiPixelDynamicInefficiencyColGeomFactorMap = SiPixelDynamicInefficiencyFullPixelMap<SiPixDynIneff::colgeom>;
  using SiPixelDynamicInefficiencyChipGeomFactorMap = SiPixelDynamicInefficiencyFullPixelMap<SiPixDynIneff::chipgeom>;

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
  PAYLOAD_INSPECTOR_CLASS(SiPixelDynamicInefficiencyGeomFactorMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelDynamicInefficiencyColGeomFactorMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelDynamicInefficiencyChipGeomFactorMap);
}
