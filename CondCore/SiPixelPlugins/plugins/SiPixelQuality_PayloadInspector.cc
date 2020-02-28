/*!
  \file SiPixelQulity_PayloadInspector
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

  /************************************************
    test class
  *************************************************/

  class SiPixelQualityTest : public cond::payloadInspector::Histogram1D<SiPixelQuality> {
  public:
    SiPixelQualityTest()
        : cond::payloadInspector::Histogram1D<SiPixelQuality>(
              "SiPixelQuality test", "SiPixelQuality test", 10, 0.0, 10.0) {
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto const& iov : iovs) {
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

  class SiPixelQualityBadRocsSummary : public cond::payloadInspector::PlotImage<SiPixelQuality> {
  public:
    SiPixelQualityBadRocsSummary() : cond::payloadInspector::PlotImage<SiPixelQuality>("SiPixel Quality Summary") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      std::vector<std::tuple<cond::Time_t, cond::Hash> > sorted_iovs = iovs;

      for (const auto& iov : iovs) {
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
   occupancy style map BPix
  *************************************************/

  class SiPixelBPixQualityMap : public cond::payloadInspector::PlotImage<SiPixelQuality> {
  public:
    SiPixelBPixQualityMap()
        : cond::payloadInspector::PlotImage<SiPixelQuality>("SiPixelQuality Barrel Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));

      static const int n_layers = 4;
      int nlad_list[n_layers] = {6, 14, 22, 32};
      int divide_roc = 1;

      // ---------------------    BOOK HISTOGRAMS
      std::array<TH2D*, n_layers> h_bpix_occ;

      for (unsigned int lay = 1; lay <= 4; lay++) {
        int nlad = nlad_list[lay - 1];

        std::string name = "occ_Layer_" + std::to_string(lay);
        std::string title = "; Module # ; Ladder #";
        h_bpix_occ[lay - 1] = new TH2D(name.c_str(),
                                       title.c_str(),
                                       72 * divide_roc,
                                       -4.5,
                                       4.5,
                                       (nlad * 4 + 2) * divide_roc,
                                       -nlad - 0.5,
                                       nlad + 0.5);
      }

      auto theDisabledModules = payload->getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        int coded_badRocs = mod.BadRocs;
        int subid = DetId(mod.DetID).subdetId();
        std::bitset<16> bad_rocs(coded_badRocs);
        if (subid == PixelSubdetector::PixelBarrel) {
          auto layer = m_trackerTopo.pxbLayer(DetId(mod.DetID));
          auto s_ladder = SiPixelPI::signed_ladder(DetId(mod.DetID), m_trackerTopo, true);
          auto s_module = SiPixelPI::signed_module(DetId(mod.DetID), m_trackerTopo, true);

          bool isFlipped = SiPixelPI::isBPixOuterLadder(DetId(mod.DetID), m_trackerTopo, false);
          if ((layer > 1 && s_module < 0))
            isFlipped = !isFlipped;

          auto ladder = m_trackerTopo.pxbLadder(DetId(mod.DetID));
          auto module = m_trackerTopo.pxbModule(DetId(mod.DetID));
          COUT << "layer:" << layer << " ladder:" << ladder << " module:" << module << " signed ladder: " << s_ladder
               << " signed module: " << s_module << std::endl;

          if (payload->IsModuleBad(mod.DetID)) {
            auto rocsToMask = SiPixelPI::maskedBarrelRocsToBins(layer, s_ladder, s_module);
            for (const auto& bin : rocsToMask) {
              h_bpix_occ[layer - 1]->SetBinContent(bin.first, bin.second, 1);
            }
          } else {
            auto rocsToMask = SiPixelPI::maskedBarrelRocsToBins(layer, s_ladder, s_module, bad_rocs, isFlipped);
            for (const auto& bin : rocsToMask) {
              h_bpix_occ[layer - 1]->SetBinContent(std::get<0>(bin), std::get<1>(bin), 1);
            }
          }
        }
      }

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 1200);
      canvas.Divide(2, 2);
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      for (unsigned int lay = 1; lay <= 4; lay++) {
        SiPixelPI::dress_occup_plot(canvas, h_bpix_occ[lay - 1], lay, 0, 1);
      }

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      for (unsigned int lay = 1; lay <= 4; lay++) {
        canvas.cd(lay);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.055);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outBPix.root");
#endif

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
   occupancy style map FPix
  *************************************************/

  class SiPixelFPixQualityMap : public cond::payloadInspector::PlotImage<SiPixelQuality> {
  public:
    SiPixelFPixQualityMap()
        : cond::payloadInspector::PlotImage<SiPixelQuality>("SiPixelQuality Forward Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));

      static const int n_rings = 2;
      std::array<TH2D*, n_rings> h_fpix_occ;
      int divide_roc = 1;

      // ---------------------    BOOK HISTOGRAMS
      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        int n = ring == 1 ? 92 : 140;
        float y = ring == 1 ? 11.5 : 17.5;
        std::string name = "occ_ring_" + std::to_string(ring);
        std::string title = "; Disk # ; Blade/Panel #";

        h_fpix_occ[ring - 1] = new TH2D(name.c_str(), title.c_str(), 56 * divide_roc, -3.5, 3.5, n * divide_roc, -y, y);
      }

      auto theDisabledModules = payload->getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        int coded_badRocs = mod.BadRocs;
        int subid = DetId(mod.DetID).subdetId();
        std::bitset<16> bad_rocs(coded_badRocs);
        if (subid == PixelSubdetector::PixelEndcap) {
          auto ring = SiPixelPI::ring(DetId(mod.DetID), m_trackerTopo, true);
          auto s_blade = SiPixelPI::signed_blade(DetId(mod.DetID), m_trackerTopo, true);
          auto s_disk = SiPixelPI::signed_disk(DetId(mod.DetID), m_trackerTopo, true);
          auto s_blade_panel = SiPixelPI::signed_blade_panel(DetId(mod.DetID), m_trackerTopo, true);
          auto panel = m_trackerTopo.pxfPanel(mod.DetID);

          //bool isFlipped = (s_disk > 0) ? (std::abs(s_blade)%2==0) : (std::abs(s_blade)%2==1);
          bool isFlipped = (s_disk > 0) ? (panel == 1) : (panel == 2);

          COUT << "ring:" << ring << " blade: " << s_blade << " panel: " << panel
               << " signed blade/panel: " << s_blade_panel << " disk: " << s_disk << std::endl;

          if (payload->IsModuleBad(mod.DetID)) {
            auto rocsToMask = SiPixelPI::maskedForwardRocsToBins(ring, s_blade, panel, s_disk);
            for (const auto& bin : rocsToMask) {
              h_fpix_occ[ring - 1]->SetBinContent(bin.first, bin.second, 1);
            }
          } else {
            auto rocsToMask = SiPixelPI::maskedForwardRocsToBins(ring, s_blade, panel, s_disk, bad_rocs, isFlipped);
            for (const auto& bin : rocsToMask) {
              h_fpix_occ[ring - 1]->SetBinContent(std::get<0>(bin), std::get<1>(bin), 1);
            }
          }
        }  // if it's endcap
      }    // loop on disable moduels

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 600);
      canvas.Divide(2, 1);
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        SiPixelPI::dress_occup_plot(canvas, h_fpix_occ[ring - 1], 0, ring, 1);
      }

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        canvas.cd(ring);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.050);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outFPix.root");
#endif
      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQuality) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsSummary);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsTimeHistory);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixQualityMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixQualityMap);
}
