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

#include <cmath>
#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TCanvas.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "TF1.h"
#include "TMath.h"
#include <Math/Polynomial.h>

namespace {

  using namespace cond::payloadInspector;
  namespace SiPixDynIneff {

    // different types of geometrical inefficiency factors
    enum factor { geom = 0, colgeom = 1, chipgeom = 2, pu = 3, INVALID = 4 };
    const std::array<std::string, 5> factorString = {
        {"pixel geometry", "column geometry", "chip geometry", "PU", "invalid"}};

    using FactorMap = std::map<unsigned int, double>;
    using PUFactorMap = std::map<unsigned int, std::vector<double> >;

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
      for (const auto& map_element : map_pufactory) {
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

    //(Not used for the moment)
    //_________________________________________________
    [[maybe_unused]] bool matches(const DetId& detid, const DetId& db_id, const std::vector<uint32_t>& DetIdmasks) {
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
    bool checkPhase(const SiPixelPI::phase phase, const std::vector<uint32_t>& masks_db) {
      const char* inputFile;
      switch (phase) {
        case SiPixelPI::phase::zero:
          inputFile = "Geometry/TrackerCommonData/data/trackerParameters.xml";
          break;
        case SiPixelPI::phase::one:
          inputFile = "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
          break;
        case SiPixelPI::phase::two:
          inputFile = "Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml";
          break;
        default:
          throw cms::Exception("SiPixelDynamicInefficiency_PayloadInspector") << "checkPhase: unrecongnized phase!";
      }

      // create the standalone tracker topology
      const auto& tkTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(inputFile).fullPath());

      // Check what masks we would get using the current geometry
      // It has to match what is in the db content!!

      std::vector<uint32_t> masks_geom;
      uint32_t max = std::numeric_limits<uint32_t>::max();

      masks_geom.push_back(tkTopo.pxbDetId(max, 0, 0).rawId());
      masks_geom.push_back(tkTopo.pxbDetId(0, max, 0).rawId());
      masks_geom.push_back(tkTopo.pxbDetId(0, 0, max).rawId());
      masks_geom.push_back(tkTopo.pxfDetId(max, 0, 0, 0, 0).rawId());
      masks_geom.push_back(tkTopo.pxfDetId(0, max, 0, 0, 0).rawId());
      masks_geom.push_back(tkTopo.pxfDetId(0, 0, max, 0, 0).rawId());
      masks_geom.push_back(tkTopo.pxfDetId(0, 0, 0, max, 0).rawId());
      masks_geom.push_back(tkTopo.pxfDetId(0, 0, 0, 0, max).rawId());

      return (masks_geom.size() == masks_db.size() &&
              std::equal(masks_geom.begin(), masks_geom.end(), masks_db.begin()));
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
   Full Pixel Tracker Map class (for geometrical factors)
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

        if (!SiPixDynIneff::checkPhase(SiPixelPI::phase::one, detIdmasks_db)) {
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

  /************************************************
   Full Pixel Tracker Map class (for PU factors)
  *************************************************/
  class SiPixelDynamicInefficiencyPUPixelMaps : public PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV> {
  public:
    SiPixelDynamicInefficiencyPUPixelMaps()
        : PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV>("SiPixelDynamicInefficiency Map") {
      label_ = "SiPixelDynamicInefficiencyFullPixelMap";
      payloadString = fmt::sprintf("%s Dynamic Inefficiency", SiPixDynIneff::factorString[SiPixDynIneff::pu]);
    }

    bool fill() override {
      gStyle->SetPalette(1);
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelDynamicInefficiency> payload = this->fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        std::vector<Phase1PixelSummaryMap> maps;

        SiPixDynIneff::PUFactorMap theMap = payload->getPUFactors();
        std::vector<uint32_t> detIdmasks_db = payload->getDetIdmasks();

        if (!SiPixDynIneff::checkPhase(SiPixelPI::phase::one, detIdmasks_db)) {
          edm::LogError(label_) << label_ << " maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, 0);
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        }

        unsigned int depth = maxDepthOfPUArray(theMap);

        // create the maps
        for (unsigned int i = 0; i < depth; i++) {
          maps.emplace_back(
              "", fmt::sprintf("%s, factor %i", payloadString, i), fmt::sprintf("%s, factor %i", payloadString, i));
          maps[i].createTrackerBaseMap();
        }

        // retrieve the list of phase1 detids
        const auto& reader =
            SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh1DefaultFile).fullPath());
        const auto& p1detIds = reader.getAllDetIds();

        // fill the maps
        for (const auto& det : p1detIds) {
          const auto& values = SiPixDynIneff::getMatchingPUFactors(det, theMap, detIdmasks_db);
          int index = 0;
          for (const auto& value : values) {
            maps[index].fillTrackerMap(det, value);
            index++;
          }
        }

        // in case the map is completely filled with one value;
        // set the z-axis to be meaningful
        for (unsigned int i = 0; i < depth; i++) {
          const auto& range = maps[i].getZAxisRange();
          if (range.first == range.second) {
            maps[i].setZAxisRange(range.first - 0.01, range.second + 0.01);
          }
        }

        // determine how the plot will be paginated
        auto sides = getClosestFactors(depth);
        TCanvas canvas("Canv", "Canv", sides.second * 900, sides.first * 600);
        canvas.Divide(sides.second, sides.first);

        // print the sub-canvases
        for (unsigned int i = 0; i < depth; i++) {
          maps[i].printTrackerMap(canvas, 0.035, i + 1);
          auto ltx = TLatex();
          ltx.SetTextFont(62);
          ltx.SetTextSize(0.025);
          ltx.SetTextAlign(11);
          ltx.DrawLatexNDC(
              gPad->GetLeftMargin() + 0.01,
              gPad->GetBottomMargin() + 0.01,
              ("#color[4]{" + tag.name + "}, IOV: #color[4]{" + std::to_string(std::get<0>(iov)) + "}").c_str());
        }

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

  protected:
    std::string payloadString;
    std::string label_;

  private:
    unsigned int maxDepthOfPUArray(const std::map<unsigned int, std::vector<double> >& map_pufactor) {
      unsigned int size{0};
      for (const auto& [id, vec] : map_pufactor) {
        if (vec.size() > size)
          size = vec.size();
      }
      return size;
    }

    std::pair<int, int> getClosestFactors(int input) {
      if ((input % 2 != 0) && input > 1) {
        input += 1;
      }

      int testNum = (int)sqrt(input);
      while (input % testNum != 0) {
        testNum--;
      }
      return std::make_pair(testNum, input / testNum);
    }
  };

  /************************************************
   Per sector plot of the inefficiency parameterization vs inst lumi (for PU factors)
  *************************************************/
  class SiPixelDynamicInefficiencyPUParametrization : public PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV> {
  public:
    SiPixelDynamicInefficiencyPUParametrization()
        : PlotImage<SiPixelDynamicInefficiency, SINGLE_IOV>("SiPixelDynamicInefficiency PU parametrization") {
      label_ = "SiPixelDynamicInefficiencyPUParameterization";
      payloadString =
          fmt::sprintf("%s Dynamic Inefficiency parametrization", SiPixDynIneff::factorString[SiPixDynIneff::pu]);
    }

    bool fill() override {
      gStyle->SetPalette(1);
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelDynamicInefficiency> payload = this->fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        SiPixDynIneff::PUFactorMap theMap = payload->getPUFactors();
        std::vector<uint32_t> detIdmasks_db = payload->getDetIdmasks();

        if (!SiPixDynIneff::checkPhase(SiPixelPI::phase::one, detIdmasks_db)) {
          edm::LogError(label_) << label_ << " maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, 0);
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        }

        std::vector<std::vector<double> > listOfParametrizations;

        // retrieve the list of phase1 detids
        const auto& reader =
            SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh1DefaultFile).fullPath());
        auto p1detIds = reader.getAllDetIds();

        // follows hack to get an inner ladder module first
        // skip the first 8 dets since they lay on the same ladder
        std::swap(p1detIds[0], p1detIds[8]);

        std::map<unsigned int, std::vector<uint32_t> > modules_per_region;

        // fill the maps
        for (const auto& det : p1detIds) {
          const auto& values = SiPixDynIneff::getMatchingPUFactors(det, theMap, detIdmasks_db);
          // find the index in the vector
          auto pIndex = std::find(listOfParametrizations.begin(), listOfParametrizations.end(), values);
          // if it's not there push it back in the list of parametrizations and then insert in the map
          if (pIndex == listOfParametrizations.end()) {
            listOfParametrizations.push_back(values);
            std::vector<unsigned int> toInsert = {det};
            modules_per_region.insert(std::make_pair(listOfParametrizations.size() - 1, toInsert));
          } else {
            modules_per_region.at(pIndex - listOfParametrizations.begin()).push_back(det);
          }
        }

        unsigned int depth = listOfParametrizations.size();

        std::vector<std::string> namesOfParts;
        namesOfParts.reserve(depth);
        // fill in the (ordered) information about regions
        for (const auto& [index, modules] : modules_per_region) {
          auto PhInfo = SiPixelPI::PhaseInfo(SiPixelPI::phase1size);
          const auto& regName = this->attachLocationLabel(modules, PhInfo);
          namesOfParts.push_back(regName);

          /*
	   *  The following is needed for internal
	   *  debug / cross-check
	   */

          std::stringstream ss;
          ss << "region name: " << regName << " has the following modules attached: [";
          for (const auto& module : modules) {
            ss << module << ", ";
          }
          ss.seekp(-2, std::ios_base::end);  // remove last two chars
          ss << "] "
             << " has representation (";
          for (const auto& param : listOfParametrizations[index]) {
            ss << param << ", ";
          }
          ss.seekp(-2, std::ios_base::end);  // remove last two chars
          ss << ") ";
          edm::LogPrint(label_) << ss.str() << "\n\n";
          ss.str(std::string()); /* clear the stringstream */
        }

        // functional for polynomial of n-th degree
        auto func = [](double* x, double* p) {
          int n = p[0];
          double* params = p + 1;
          ROOT::Math::Polynomial pol(n);
          return pol(x, params);
        };

        std::vector<TF1*> parametrizations;
        std::vector<std::string> formulas;
        parametrizations.reserve(depth);
        formulas.reserve(depth);

        int index{0};
        for (auto& params : listOfParametrizations) {
          index++;
          int n = params.size();
          int npar = n + 2;
          std::string str{namesOfParts[index - 1]};
          if (str.length() >= 2 && str.substr(str.length() - 2) == "/i") {
            str += "nner";
          } else if (str.length() >= 2 && str.substr(str.length() - 2) == "/o") {
            str += "uter";
          }

          TF1* f1 = new TF1((fmt::sprintf("region: #bf{%s}", str)).c_str(), func, xmin_, xmax_, npar);

          // push polynomial degree as first entry in the vector
          params.insert(params.begin(), n);
          // TF1::SetParameters needs a C-style array
          double* arr = params.data();
          f1->SetLineWidth(2);

          // fill in the parameter
          for (unsigned int j = 0; j < params.size(); j++) {
            f1->SetParameter(j, arr[j]);
          }

          parametrizations.push_back(f1);

          // build the formula to be displayed
          std::string formula;
          edm::LogVerbatim(label_) << "index: " << index;
          for (unsigned int i = 1; i < params.size(); i++) {
            edm::LogVerbatim(label_) << " " << params[i];
            formula +=
                fmt::sprintf("%s%fx^{%i}", (i == 1 ? "" : (std::signbit(params[i]) ? "" : "+")), params[i], i - 1);
          }
          edm::LogVerbatim(label_) << std::endl;
          formulas.push_back(formula);
        }

        // determine how the plot will be paginated
        auto sides = getClosestFactors(depth);
        TCanvas canvas("Canv", "Canv", sides.second * 900, sides.first * 600);
        canvas.Divide(sides.second, sides.first);

        // print the sub-canvases
        for (unsigned int i = 0; i < depth; i++) {
          canvas.cd(i + 1);
          gPad->SetGrid();
          SiPixelPI::adjustCanvasMargins(gPad, 0.07, 0.14, 0.14, 0.03);
          parametrizations[i]->Draw();

          // set axis
          TH1* h = parametrizations[i]->GetHistogram();
          TAxis* ax = h->GetXaxis();
          ax->SetTitle("Inst. luminosity [10^{33} cm^{-2}s^{-1}]");

          TAxis* ay = h->GetYaxis();
          ay->SetRangeUser(0.80, 1.00);  // force the scale
          ay->SetTitle("Double Column Efficiency parametrization");

          // beautify
          SiPixelPI::makeNicePlotStyle(h);

          auto ltx = TLatex();
          ltx.SetTextFont(62);
          ltx.SetTextSize(0.045);
          ltx.SetTextAlign(11);
          ltx.DrawLatexNDC(
              gPad->GetLeftMargin() + 0.03,
              gPad->GetBottomMargin() + 0.04,
              ("#color[4]{" + tag.name + "}, IOV: #color[4]{" + std::to_string(std::get<0>(iov)) + "}").c_str());

          auto ltxForm = TLatex();
          ltxForm.SetTextFont(62);
          ltxForm.SetTextColor(kRed);
          ltxForm.SetTextSize(0.045);
          ltxForm.SetTextAlign(11);
          ltxForm.DrawLatexNDC(gPad->GetLeftMargin() + 0.05, 0.85, formulas[i].c_str());

          edm::LogPrint(label_) << namesOfParts[i] << " => " << formulas[i] << std::endl;
        }

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

      }  // if payload.get()
      return true;
    }  // fill()

  protected:
    std::string payloadString;
    std::string label_;

    static constexpr double xmin_ = 0.;   // x10e34 (min inst. lumi)
    static constexpr double xmax_ = 25.;  // x10e34 (max inst. lumi)

  private:
    unsigned int maxDepthOfPUArray(const std::map<unsigned int, std::vector<double> >& map_pufactor) {
      unsigned int size{0};
      for (const auto& [id, vec] : map_pufactor) {
        if (vec.size() > size)
          size = vec.size();
      }
      return size;
    }

    std::pair<int, int> getClosestFactors(int input) {
      if ((input % 2 != 0) && input > 1) {
        input += 1;
      }

      int testNum = (int)sqrt(input);
      while (input % testNum != 0) {
        testNum--;
      }
      return std::make_pair(testNum, input / testNum);
    }

    /** This function determines from the list of attached DetId which
     *  SiPixelPI::region represents them
     */
    std::string attachLocationLabel(const std::vector<uint32_t>& listOfDetIds, SiPixelPI::PhaseInfo& phInfo) {
      // collect all the regions in which it can be split
      std::vector<std::string> regions;
      for (const auto& rawId : listOfDetIds) {
        SiPixelPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(rawId);

        const char* path_toTopologyXML = phInfo.pathToTopoXML();
        auto tTopo =
            StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());
        t_info_fromXML.fillGeometryInfo(detid, tTopo, phInfo.phase());
        const auto& reg = SiPixelPI::getStringFromRegionEnum(t_info_fromXML.filterThePartition());
        if (!std::count(regions.begin(), regions.end(), reg)) {
          regions.push_back(reg);
        }
      }

      std::string retVal = "";
      // if perfect match (only one category)
      if (regions.size() == 1) {
        retVal = regions.front();
      } else {
        retVal = this->findStem(regions);
      }

      // if the last char is "/" strip it from the string
      if (retVal.back() == '/')
        retVal.pop_back();

      return retVal;
    }

    /** Given an input list of std::string this
     *  finds the longest common substring
     */
    std::string findStem(const std::vector<std::string>& arr) {
      // Determine size of the array
      int n = arr.size();

      // Take first word from array as reference
      const std::string& s = arr[0];
      int len = s.length();

      std::string res = "";

      for (int i = 0; i < len; i++) {
        for (int j = i + 1; j <= len; j++) {
          // generating all possible substrings
          // of our reference string arr[0] i.e s
          std::string stem = s.substr(i, j);
          int k = 1;
          for (k = 1; k < n; k++) {
            // Check if the generated stem is
            // common to all words
            if (arr[k].find(stem) == std::string::npos)
              break;
          }

          // If current substring is present in
          // all strings and its length is greater
          // than current result
          if (k == n && res.length() < stem.length())
            res = stem;
        }
      }
      return res;
    }
  };
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
  PAYLOAD_INSPECTOR_CLASS(SiPixelDynamicInefficiencyPUPixelMaps);
  PAYLOAD_INSPECTOR_CLASS(SiPixelDynamicInefficiencyPUParametrization);
}
