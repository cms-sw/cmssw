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
            std::cout << "detId:" << mod.DetID << " error type:" << mod.errorType << " BadRocs:" << BadRocCount
                      << std::endl;
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
        auto unpacked = unpack(std::get<0>(iov));

        std::cout << "======================= " << unpacked.first << " : " << unpacked.second << std::endl;
        auto theDisabledModules = payload->getBadComponentList();
        for (const auto& mod : theDisabledModules) {
          std::cout << "detId: " << mod.DetID << " |error type: " << mod.errorType << " |BadRocs: " << mod.BadRocs
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

    std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
      auto kLowMask = 0XFFFFFFFF;
      auto run = (since >> 32);
      auto lumi = (since & kLowMask);
      return std::make_pair(run, lumi);
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
        if (payload->IsModuleBad(mod.DetID)) {
          int subid = DetId(mod.DetID).subdetId();
          if (subid == PixelSubdetector::PixelBarrel) {
            auto layer = m_trackerTopo.pxbLayer(DetId(mod.DetID));
            auto s_ladder = SiPixelPI::signed_ladder(DetId(mod.DetID), m_trackerTopo, true);
            auto s_module = SiPixelPI::signed_module(DetId(mod.DetID), m_trackerTopo, true);

            //auto ladder = m_trackerTopo.pxbLadder(DetId(mod.DetID));
            //auto module = m_trackerTopo.pxbModule(DetId(mod.DetID));
            // std::cout <<"layer:" << layer << " ladder:" << ladder << " module:" << module
            //	         <<" signed ladder: "<< s_ladder
            //           <<" signed module: "<< s_module << std::endl;

            std::vector<std::pair<int, int> > rocsToMask = maskedBarrelRocsToBins(layer, s_ladder, s_module);
            for (const auto& bin : rocsToMask) {
              h_bpix_occ[layer - 1]->SetBinContent(bin.first, bin.second, 1);
            }
          }
        }
        std::bitset<16> bad_rocs(coded_badRocs);
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
        ltx.SetTextSize(0.06);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    // #============================================================================
    std::vector<std::pair<int, int> > maskedBarrelRocsToBins(int layer, int ladder, int module) {
      std::vector<std::pair<int, int> > rocsToMask;

      int nlad_list[4] = {6, 14, 22, 32};
      int nlad = nlad_list[layer - 1];

      int start_x = module > 0 ? ((module + 4) * 8) + 1 : ((4 - (std::abs(module))) * 8) + 1;
      int start_y = ladder > 0 ? ((ladder + nlad) * 2) + 1 : ((nlad - (std::abs(ladder))) * 2) + 1;

      int end_x = start_x + 7;
      int end_y = start_y + 1;

      std::cout << "module: " << module << " start_x:" << start_x << " end_x:" << end_x << std::endl;
      std::cout << "ladder: " << ladder << " start_y:" << start_y << " end_y:" << end_y << std::endl;
      std::cout << "==================================================================" << std::endl;

      for (int bin_x = 1; bin_x <= 72; bin_x++) {
        for (int bin_y = 1; bin_y <= (nlad * 4 + 2); bin_y++) {
          if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
            rocsToMask.push_back(std::make_pair(bin_x, bin_y));
          }
        }
      }

      return rocsToMask;
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
        if (payload->IsModuleBad(mod.DetID)) {
          int subid = DetId(mod.DetID).subdetId();
          if (subid == PixelSubdetector::PixelEndcap) {
            auto ring = SiPixelPI::ring(DetId(mod.DetID), m_trackerTopo, true);
            auto s_blade = SiPixelPI::signed_blade(DetId(mod.DetID), m_trackerTopo, true);
            auto s_disk = SiPixelPI::signed_disk(DetId(mod.DetID), m_trackerTopo, true);
            auto s_blade_panel = SiPixelPI::signed_blade_panel(DetId(mod.DetID), m_trackerTopo, true);
            auto panel = m_trackerTopo.pxfPanel(mod.DetID);

            std::cout << "ring:" << ring << " blade: " << s_blade << " panel: " << panel << " disk: " << s_disk
                      << std::endl;

            std::vector<std::pair<int, int> > rocsToMask = maskedForwardRocsToBins(ring, s_blade, panel, s_disk);
            for (const auto& bin : rocsToMask) {
              h_fpix_occ[ring - 1]->SetBinContent(bin.first, bin.second, 1);
            }
          }
        }
        std::bitset<16> bad_rocs(coded_badRocs);
      }

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
        ltx.SetTextSize(0.06);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      canvas.SaveAs("out.root");

      return true;
    }

    // #============================================================================
    std::vector<std::pair<int, int> > maskedForwardRocsToBins(int ring, int blade, int panel, int disk) {
      std::vector<std::pair<int, int> > rocsToMask;

      int nblade_list[2] = {11, 17};
      int nybins_list[2] = {92, 140};
      int nblade = nblade_list[ring - 1];
      int nybins = nybins_list[ring - 1];

      int start_x = disk > 0 ? ((disk + 3) * 8) + 1 : ((3 - (std::abs(disk))) * 8) + 1;
      //int start_y = blade > 0 ? ((blade+nblade)*4)-panel*2  : ((nblade-(std::abs(blade)))*4)-panel*2;
      int start_y = blade > 0 ? (nybins / 2) + (blade * 4) - (panel * 2) + 3
                              : ((nybins / 2) - (std::abs(blade) * 4) - panel * 2) + 3;

      int end_x = start_x + 7;
      int end_y = start_y + 1;

      std::cout << "==================================================================" << std::endl;
      std::cout << "disk:  " << disk << " start_x:" << start_x << " end_x:" << end_x << std::endl;
      std::cout << "blade: " << blade << " start_y:" << start_y << " end_y:" << end_y << std::endl;

      for (int bin_x = 1; bin_x <= 56; bin_x++) {
        for (int bin_y = 1; bin_y <= nybins; bin_y++) {
          if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
            rocsToMask.push_back(std::make_pair(bin_x, bin_y));
          }
        }
      }

      return rocsToMask;
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
