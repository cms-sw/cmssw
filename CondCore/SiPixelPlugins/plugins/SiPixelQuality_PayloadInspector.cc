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

// #define MMDEBUG
#ifdef MMDEBUG
#include <iostream>
#define COUT std::cout << "MM "
#else
#define COUT edm::LogVerbatim("")
#endif

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
            auto rocsToMask = maskedBarrelRocsToBins(layer, s_ladder, s_module);
            for (const auto& bin : rocsToMask) {
              h_bpix_occ[layer - 1]->SetBinContent(bin.first, bin.second, 1);
            }
          } else {
            auto rocsToMask = maskedBarrelRocsToBins(layer, s_ladder, s_module, bad_rocs, isFlipped);
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
        ltx.SetTextSize(0.06);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outBPix.root");
#endif

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

      COUT << "module: " << module << " start_x:" << start_x << " end_x:" << end_x << std::endl;
      COUT << "ladder: " << ladder << " start_y:" << start_y << " end_y:" << end_y << std::endl;
      COUT << "==================================================================" << std::endl;

      for (int bin_x = 1; bin_x <= 72; bin_x++) {
        for (int bin_y = 1; bin_y <= (nlad * 4 + 2); bin_y++) {
          if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
            rocsToMask.push_back(std::make_pair(bin_x, bin_y));
          }
        }
      }
      return rocsToMask;
    }

    // #============================================================================
    std::vector<std::tuple<int, int, int> > maskedBarrelRocsToBins(
        int layer, int ladder, int module, std::bitset<16> bad_rocs, bool isFlipped) {
      std::vector<std::tuple<int, int, int> > rocsToMask;

      int nlad_list[4] = {6, 14, 22, 32};
      int nlad = nlad_list[layer - 1];

      int start_x = module > 0 ? ((module + 4) * 8) + 1 : ((4 - (std::abs(module))) * 8) + 1;
      int start_y = ladder > 0 ? ((ladder + nlad) * 2) + 1 : ((nlad - (std::abs(ladder))) * 2) + 1;

      int roc0_x = ((layer == 1) || (layer > 1 && module > 0)) ? start_x + 7 : start_x;
      int roc0_y = start_y - 1;

      size_t idx = 0;
      while (idx < bad_rocs.size()) {
        if (bad_rocs.test(idx)) {
          //////////////////////////////////////////////////////////////////////////////////////
          //		                            |					      //
          // In BPix Layer1 and module>0 in L2,3,4  |   In BPix Layer 2,3,4 module > 0	      //
          //                                        |					      //
          // ROCs are ordered in the following      |   ROCs are ordered in the following     //
          // fashion for unplipped modules 	    |   fashion for unplipped modules         //
          //					    |  				              //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 8 |9  |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          //					    |  					      //
          // if the module is flipped the ordering  |   if the module is flipped the ordering //
          // is reveresed                           |   is reversed                           //
          //					    |                                         //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 8 | 9 |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          //////////////////////////////////////////////////////////////////////////////////////

          int roc_x(0), roc_y(0);

          if ((layer == 1) || (layer > 1 && module > 0)) {
            if (!isFlipped) {
              roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
              roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
            } else {
              roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
              roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
            }
          } else {
            if (!isFlipped) {
              roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
              roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
            } else {
              roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
              roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
            }
          }

          COUT << bad_rocs << " : (idx)= " << idx << std::endl;
          COUT << " layer:  " << layer << std::endl;
          COUT << "module: " << module << " roc_x:" << roc_x << std::endl;
          COUT << "ladder: " << ladder << " roc_y:" << roc_y << std::endl;
          COUT << "==================================================================" << std::endl;

          rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
        }
        ++idx;
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
            auto rocsToMask = maskedForwardRocsToBins(ring, s_blade, panel, s_disk);
            for (const auto& bin : rocsToMask) {
              h_fpix_occ[ring - 1]->SetBinContent(bin.first, bin.second, 1);
            }
          } else {
            auto rocsToMask = maskedForwardRocsToBins(ring, s_blade, panel, s_disk, bad_rocs, isFlipped);
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
        ltx.SetTextSize(0.06);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outFPix.root");
#endif
      return true;
    }

    // #============================================================================
    std::vector<std::pair<int, int> > maskedForwardRocsToBins(int ring, int blade, int panel, int disk) {
      std::vector<std::pair<int, int> > rocsToMask;

      //int nblade_list[2] = {11, 17};
      int nybins_list[2] = {92, 140};
      //int nblade = nblade_list[ring - 1];
      int nybins = nybins_list[ring - 1];

      int start_x = disk > 0 ? ((disk + 3) * 8) + 1 : ((3 - (std::abs(disk))) * 8) + 1;
      //int start_y = blade > 0 ? ((blade+nblade)*4)-panel*2  : ((nblade-(std::abs(blade)))*4)-panel*2;
      int start_y = blade > 0 ? (nybins / 2) + (blade * 4) - (panel * 2) + 3
                              : ((nybins / 2) - (std::abs(blade) * 4) - panel * 2) + 3;

      int end_x = start_x + 7;
      int end_y = start_y + 1;

      COUT << "==================================================================" << std::endl;
      COUT << "disk:  " << disk << " start_x:" << start_x << " end_x:" << end_x << std::endl;
      COUT << "blade: " << blade << " start_y:" << start_y << " end_y:" << end_y << std::endl;

      for (int bin_x = 1; bin_x <= 56; bin_x++) {
        for (int bin_y = 1; bin_y <= nybins; bin_y++) {
          if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
            rocsToMask.push_back(std::make_pair(bin_x, bin_y));
          }
        }
      }
      return rocsToMask;
    }

    // #============================================================================
    std::vector<std::tuple<int, int, int> > maskedForwardRocsToBins(
        int ring, int blade, int panel, int disk, std::bitset<16> bad_rocs, bool isFlipped) {
      std::vector<std::tuple<int, int, int> > rocsToMask;

      //int nblade_list[2] = {11, 17};
      int nybins_list[2] = {92, 140};
      //int nblade = nblade_list[ring - 1];
      int nybins = nybins_list[ring - 1];

      int start_x = disk > 0 ? ((disk + 3) * 8) + 1 : ((3 - (std::abs(disk))) * 8) + 1;
      //int start_y = blade > 0 ? ((blade+nblade)*4)-panel*2  : ((nblade-(std::abs(blade)))*4)-panel*2;
      int start_y = blade > 0 ? (nybins / 2) + (blade * 4) - (panel * 2) + 3
                              : ((nybins / 2) - (std::abs(blade) * 4) - panel * 2) + 3;

      int roc0_x = disk > 0 ? start_x + 7 : start_x;
      int roc0_y = start_y - 1;

      size_t idx = 0;
      while (idx < bad_rocs.size()) {
        if (bad_rocs.test(idx)) {
          int roc_x(0), roc_y(0);

          //////////////////////////////////////////////////////////////////////////////////////
          //		                            |					      //
          // In FPix + (Disk 1,2,3)                 |   In FPix - (Disk -1,-2,-3)	      //
          //                                        |					      //
          // ROCs are ordered in the following      |   ROCs are ordered in the following     //
          // fashion for unplipped modules 	    |   fashion for unplipped modules         //
          //					    |  				              //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 8 |9  |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          //					    |  					      //
          // if the module is flipped the ordering  |   if the module is flipped the ordering //
          // is reveresed                           |   is reversed                           //
          //					    |                                         //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          // | 8 | 9 |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
          // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
          //////////////////////////////////////////////////////////////////////////////////////

          if (disk > 0) {
            if (!isFlipped) {
              roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
              roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
            } else {
              roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
              roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
            }
          } else {
            if (!isFlipped) {
              roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
              roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
            } else {
              roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
              roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
            }
          }

          COUT << bad_rocs << " : (idx)= " << idx << std::endl;
          COUT << " panel: " << panel << " isFlipped: " << isFlipped << std::endl;
          COUT << " disk:  " << disk << " roc_x:" << roc_x << std::endl;
          COUT << " blade: " << blade << " roc_y:" << roc_y << std::endl;
          COUT << "===============================" << std::endl;

          rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
        }
        ++idx;
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
