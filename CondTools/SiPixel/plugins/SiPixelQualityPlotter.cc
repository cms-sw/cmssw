// -*- C++ -*-
//
// Package:    CondTools/SiPixel
// Class:      SiPixelQualityPlotter
//
/**\class SiPixelQualityPlotter SiPixelQualityPlotter.cc CondTools/SiPixel/plugins/SiPixelQualityPlotter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Fri, 05 Jun 2020 12:28:51 GMT
//
//

// user include files
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// system includes
#include <iostream>
#include <fstream>  // std::ifstream
#include <memory>

// ROOT includes
#include "TH2.h"
#include "TCanvas.h"

namespace payloadPlotting {

  /*--------------------------------------------------------------------*/
  std::vector<std::tuple<int, int, int>> maskedBarrelRocsToBins(
      int layer, int ladder, int module, std::bitset<16> bad_rocs, bool isFlipped)
  /*--------------------------------------------------------------------*/
  {
    std::vector<std::tuple<int, int, int>> rocsToMask;

    int nlad_list[4] = {6, 14, 22, 32};
    int nlad = nlad_list[layer - 1];

    int start_x = module > 0 ? ((module + 4) * 8) + 1 : ((4 - (std::abs(module))) * 8) + 1;
    int start_y = ladder > 0 ? ((ladder + nlad) * 2) + 1 : ((nlad - (std::abs(ladder))) * 2) + 1;

    int roc0_x = ((layer == 1) || (layer > 1 && module > 0)) ? start_x + 7 : start_x;
    int roc0_y = start_y - 1;

    size_t idx = 0;
    while (idx < bad_rocs.size()) {
      if (bad_rocs.test(idx)) {
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

        edm::LogInfo("payloadPlotting") << bad_rocs << " : (idx)= " << idx << std::endl;
        edm::LogInfo("payloadPlotting") << " layer:  " << layer << std::endl;
        edm::LogInfo("payloadPlotting") << "module: " << module << " roc_x:" << roc_x << std::endl;
        edm::LogInfo("payloadPlotting") << "ladder: " << ladder << " roc_y:" << roc_y << std::endl;
        edm::LogInfo("payloadPlotting") << "=================================================================="
                                        << std::endl;

        rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
      }
      ++idx;
    }
    return rocsToMask;
  }

  /*--------------------------------------------------------------------*/
  std::vector<std::tuple<int, int, int>> maskedForwardRocsToBins(
      int ring, int blade, int panel, int disk, std::bitset<16> bad_rocs, bool isFlipped)
  /*--------------------------------------------------------------------*/
  {
    std::vector<std::tuple<int, int, int>> rocsToMask;
    int nybins_list[2] = {92, 140};
    int nybins = nybins_list[ring - 1];

    int start_x = disk > 0 ? ((disk + 3) * 8) + 1 : ((3 - (std::abs(disk))) * 8) + 1;
    int start_y = blade > 0 ? (nybins / 2) + (blade * 4) - (panel * 2) + 3
                            : ((nybins / 2) - (std::abs(blade) * 4) - panel * 2) + 3;

    int roc0_x = disk > 0 ? start_x + 7 : start_x;
    int roc0_y = start_y - 1;

    size_t idx = 0;
    while (idx < bad_rocs.size()) {
      if (bad_rocs.test(idx)) {
        int roc_x(0), roc_y(0);

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

        edm::LogInfo("payloadPlotting") << bad_rocs << " : (idx)= " << idx << std::endl;
        edm::LogInfo("payloadPlotting") << " panel: " << panel << " isFlipped: " << isFlipped << std::endl;
        edm::LogInfo("payloadPlotting") << " disk:  " << disk << " roc_x:" << roc_x << std::endl;
        edm::LogInfo("payloadPlotting") << " blade: " << blade << " roc_y:" << roc_y << std::endl;
        edm::LogInfo("payloadPlotting") << "===============================" << std::endl;

        rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
      }
      ++idx;
    }
    return rocsToMask;
  }
}  // namespace payloadPlotting

//
// class declaration
//

class SiPixelQualityPlotter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelQualityPlotter(const edm::ParameterSet&);
  ~SiPixelQualityPlotter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void CMS_lumi(TPad* pad, float lumi);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsToken_;
  const edm::ESGetToken<SiPixelQuality, SiPixelQualityFromDbRcd> qualEsToken_;

  const bool useLogScale_;
  const bool addLumiInfo_;
  const std::string analyzedTag_;
  const unsigned int lastRun_;

  const edm::InputTag lumiInputTag_;
  const edm::EDGetTokenT<LumiInfo> lumiToken_;

  int IOVcount_;
  edm::ESWatcher<SiPixelQualityFromDbRcd> SiPixelQualityWatcher_;

  float totalLumi_;
  float lumiSinceLastReset_;
  cond::Time_t lastIOVtime_;

  // actual histograms to fill
  static const int n_layers = 4;
  std::array<TH2D*, n_layers> h_bpix_occ;

  static const int n_rings = 2;
  std::array<TH2D*, n_rings> h_fpix_occ;

  // cached payload
  std::map<uint32_t, std::bitset<16>> cachedPayload_;
};

//
// constructors and destructor
//
SiPixelQualityPlotter::SiPixelQualityPlotter(const edm::ParameterSet& iConfig)
    : topoEsToken_(esConsumes()),
      qualEsToken_(esConsumes()),
      useLogScale_(iConfig.getParameter<bool>("useLogScale")),
      addLumiInfo_(iConfig.getParameter<bool>("addLumiInfo")),
      analyzedTag_(iConfig.getParameter<std::string>("analyzedTag")),
      lastRun_(iConfig.getUntrackedParameter<unsigned int>("maxRun", 999999)),
      lumiInputTag_(iConfig.getUntrackedParameter<edm::InputTag>("lumiInputTag")),
      lumiToken_(consumes<LumiInfo>(lumiInputTag_)) {
  // initialize the counters

  IOVcount_ = 0;
  totalLumi_ = 0.f;
  lumiSinceLastReset_ = 0.f;
  lastIOVtime_ = 0;

  // ---------------------    BOOK HISTOGRAMS
  // BPIX
  int nlad_list[n_layers] = {6, 14, 22, 32};
  int divide_roc = 1;

  for (unsigned int lay = 1; lay <= n_layers; lay++) {
    int nlad = nlad_list[lay - 1];

    std::string name = "occ_Layer_" + std::to_string(lay);
    std::string title = "; BPix Layer " + std::to_string(lay) + " Module # ; BPix Layer " + std::to_string(lay) +
                        " Ladder # ; fraction of bad luminosity per component (%)";
    h_bpix_occ[lay - 1] = new TH2D(
        name.c_str(), title.c_str(), 72 * divide_roc, -4.5, 4.5, (nlad * 4 + 2) * divide_roc, -nlad - 0.5, nlad + 0.5);
  }

  // FPIX
  for (unsigned int ring = 1; ring <= n_rings; ring++) {
    int n = ring == 1 ? 92 : 140;
    float y = ring == 1 ? 11.5 : 17.5;
    std::string name = "occ_ring_" + std::to_string(ring);
    std::string title = "; FPix Ring " + std::to_string(ring) + " Disk # ; FPix Ring " + std::to_string(ring) +
                        " Blade/Panel # ; fraction of bad luminosity per component (%)";

    h_fpix_occ[ring - 1] = new TH2D(name.c_str(), title.c_str(), 56 * divide_roc, -3.5, 3.5, n * divide_roc, -y, y);
  }
}

SiPixelQualityPlotter::~SiPixelQualityPlotter() {
  // delete the histograms
  for (unsigned int lay = 1; lay <= n_layers; lay++) {
    delete h_bpix_occ[lay - 1];
  }

  for (unsigned int ring = 1; ring <= n_rings; ring++) {
    delete h_fpix_occ[ring - 1];
  }
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelQualityPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  unsigned int RunNumber_ = iEvent.eventAuxiliary().run();
  unsigned int LuminosityBlockNumber_ = iEvent.eventAuxiliary().luminosityBlock();

  cond::UnpackedTime localtime = std::make_pair(RunNumber_, LuminosityBlockNumber_);
  cond::Time_t packedtime = cond::time::pack(localtime);

  bool hasQualityIOV = SiPixelQualityWatcher_.check(iSetup);

  const auto& m_trackerTopo = iSetup.getData(topoEsToken_);

  if (hasQualityIOV || RunNumber_ > lastRun_) {
    IOVcount_++;
    lastIOVtime_ = packedtime;

    edm::LogVerbatim("SiPixelQualityPlotter")
        << "New IOV, Run: " << RunNumber_ << " LS:" << LuminosityBlockNumber_ << std::endl;
    edm::LogVerbatim("SiPixelQualityPlotter")
        << "Accumulated Luminosity: " << lumiSinceLastReset_ << " | Total Luminosity: " << totalLumi_ << std::endl;

    for (const auto [mod, payload] : cachedPayload_) {
      int subid = DetId(mod).subdetId();
      if (subid == PixelSubdetector::PixelBarrel) {
        auto layer = m_trackerTopo.pxbLayer(DetId(mod));
        auto s_ladder = SiPixelPI::signed_ladder(DetId(mod), m_trackerTopo, true);
        auto s_module = SiPixelPI::signed_module(DetId(mod), m_trackerTopo, true);

        bool isFlipped = !SiPixelPI::isBPixOuterLadder(DetId(mod), m_trackerTopo, false);
        if ((layer > 1 && s_module < 0))
          isFlipped = !isFlipped;

        auto ladder = m_trackerTopo.pxbLadder(DetId(mod));
        auto module = m_trackerTopo.pxbModule(DetId(mod));
        LogDebug("SiPixelQualityPlotter")
            << "layer:" << layer << " ladder:" << ladder << " module:" << module << " signed ladder: " << s_ladder
            << " signed module: " << s_module << std::endl;

        auto bpix_rocsToMask = payloadPlotting::maskedBarrelRocsToBins(layer, s_ladder, s_module, payload, isFlipped);
        for (const auto& bin : bpix_rocsToMask) {
          double x = h_bpix_occ[layer - 1]->GetXaxis()->GetBinCenter(std::get<0>(bin));
          double y = h_bpix_occ[layer - 1]->GetYaxis()->GetBinCenter(std::get<1>(bin));
          h_bpix_occ[layer - 1]->Fill(x, y, lumiSinceLastReset_);
        }
      } else if (subid == PixelSubdetector::PixelEndcap) {
        auto ring = SiPixelPI::ring(DetId(mod), m_trackerTopo, true);
        auto s_blade = SiPixelPI::signed_blade(DetId(mod), m_trackerTopo, true);
        auto s_disk = SiPixelPI::signed_disk(DetId(mod), m_trackerTopo, true);
        auto s_blade_panel = SiPixelPI::signed_blade_panel(DetId(mod), m_trackerTopo, true);
        auto panel = m_trackerTopo.pxfPanel(mod);

        bool isFlipped = (s_disk > 0) ? (panel == 1) : (panel == 2);

        LogDebug("SiPixelQualityPlotter")
            << "ring:" << ring << " blade: " << s_blade << " panel: " << panel
            << " signed blade/panel: " << s_blade_panel << " disk: " << s_disk << std::endl;

        auto fpix_rocsToMask =
            payloadPlotting::maskedForwardRocsToBins(ring, s_blade, panel, s_disk, payload, isFlipped);
        for (const auto& bin : fpix_rocsToMask) {
          double x = h_fpix_occ[ring - 1]->GetXaxis()->GetBinCenter(std::get<0>(bin));
          double y = h_fpix_occ[ring - 1]->GetYaxis()->GetBinCenter(std::get<1>(bin));
          h_fpix_occ[ring - 1]->Fill(x, y, lumiSinceLastReset_);
        }
      } else {
        throw cms::Exception("LogicError") << "Unknown Pixel SubDet ID " << std::endl;
      }
    }

    // clear the chached payload from memory
    cachedPayload_.clear();

    //Retrieve the pixel quality from conditions
    const SiPixelQuality* siPixelQuality_ = &iSetup.getData(qualEsToken_);

    // cache the new payload
    auto theDisabledModules = siPixelQuality_->getBadComponentList();
    for (const auto& mod : theDisabledModules) {
      int coded_badRocs = mod.BadRocs;
      std::bitset<16> bad_rocs(coded_badRocs);
      // cache the payload
      cachedPayload_.insert(std::make_pair(mod.DetID, bad_rocs));
    }
    // reset the luminosity count to zero
    lumiSinceLastReset_ = 0;
  }  // if there has been a new IOV

  if (RunNumber_ > lastRun_)
    return;

  // retrieve the luminosity
  const LumiInfo& lumi = iEvent.get(lumiToken_);
  totalLumi_ += (lumi.integLuminosity() * 1e-9);           // convert /ub to /fb
  lumiSinceLastReset_ += (lumi.integLuminosity() * 1e-9);  // convert /ub to /fb
}

// ------------ method called once each job just before starting event loop  ------------
void SiPixelQualityPlotter::beginJob() {
  // please remove this method if not needed
}

// ------------ method called once each job just after ending the event loop  ------------
void SiPixelQualityPlotter::endJob() {
  cond::UnpackedTime unpackedtime = cond::time::unpack(lastIOVtime_);
  edm::LogVerbatim("SiPixelQualityPlotter")
      << "\n=============================================\n"
      << "Last Analyzed LS: " << unpackedtime.first << "," << unpackedtime.second << std::endl;
  edm::LogVerbatim("SiPixelQualityPlotter") << "A total of " << IOVcount_ << " IOVs have been analyzed!" << std::endl;

  gStyle->SetOptStat(0);
  //=========================
  std::vector<TCanvas*> vCanvasBarrel(n_layers);
  TCanvas canvasB("SummaryBarrel", "SummaryBarrel", 1400, 1200);
  canvasB.Divide(2, 2);
  for (unsigned int i = 1; i <= n_layers; i++) {
    canvasB.cd(i)->SetTopMargin(0.06);
    canvasB.cd(i)->SetBottomMargin(0.11);
    canvasB.cd(i)->SetLeftMargin(0.12);
    canvasB.cd(i)->SetRightMargin(0.16);
    vCanvasBarrel[i - 1] = new TCanvas(Form("Layer_%i", i), Form("Layer_%i", i), 700, 600);
    vCanvasBarrel[i - 1]->cd()->SetTopMargin(0.08);
    vCanvasBarrel[i - 1]->cd()->SetBottomMargin(0.11);
    vCanvasBarrel[i - 1]->cd()->SetLeftMargin(0.12);
    vCanvasBarrel[i - 1]->cd()->SetRightMargin(0.16);
  }

  for (unsigned int lay = 1; lay <= n_layers; lay++) {
    h_bpix_occ[lay - 1]->Scale((100. / totalLumi_));
    SiPixelPI::makeNicePlotStyle(h_bpix_occ[lay - 1]);
    h_bpix_occ[lay - 1]->GetYaxis()->SetTitleOffset(1.2);
    h_bpix_occ[lay - 1]->GetZaxis()->SetTitleOffset(1.3);
    h_bpix_occ[lay - 1]->GetZaxis()->SetTitleSize(0.042);
    h_bpix_occ[lay - 1]->GetZaxis()->CenterTitle();

    canvasB.cd(lay)->Modified();
    SiPixelPI::dress_occup_plot(canvasB, h_bpix_occ[lay - 1], lay, 0, 1, true, true, true);

    if (useLogScale_) {
      canvasB.cd(lay)->SetLogz();
    }
    if (addLumiInfo_) {
      TPad* current_pad = static_cast<TPad*>(canvasB.cd(lay));
      CMS_lumi(current_pad, totalLumi_);
    }

    vCanvasBarrel[lay - 1]->cd()->Modified();
    SiPixelPI::dress_occup_plot(*vCanvasBarrel[lay - 1], h_bpix_occ[lay - 1], lay, 0, 1, true, true, true);
    if (useLogScale_) {
      vCanvasBarrel[lay - 1]->cd()->SetLogz();
    }
    if (addLumiInfo_) {
      TPad* current_pad = static_cast<TPad*>(vCanvasBarrel[lay - 1]->cd());
      CMS_lumi(current_pad, totalLumi_);
    }
  }

  for (unsigned int lay = 1; lay <= n_layers; lay++) {
    vCanvasBarrel[lay - 1]->SaveAs(("Barrel_L" + std::to_string(lay) + "_" + analyzedTag_ + ".png").c_str());
    vCanvasBarrel[lay - 1]->SaveAs(("Barrel_L" + std::to_string(lay) + "_" + analyzedTag_ + ".pdf").c_str());
  }

  canvasB.SaveAs(("SummaryBarrel_" + analyzedTag_ + ".png").c_str());
  canvasB.SaveAs(("SummaryBarrel_" + analyzedTag_ + ".pdf").c_str());

  //=========================
  std::vector<TCanvas*> vCanvasForward(2);
  TCanvas canvasF("SummaryForward", "SummaryForward", 1400, 600);
  canvasF.Divide(2, 1);
  for (unsigned int i = 1; i <= n_rings; i++) {
    canvasF.cd(i)->SetTopMargin(0.06);
    canvasF.cd(i)->SetBottomMargin(0.11);
    canvasF.cd(i)->SetLeftMargin(0.12);
    canvasF.cd(i)->SetRightMargin(0.16);
    vCanvasForward[i - 1] = new TCanvas(Form("Ring_%i", i), Form("Ring_%i", i), 700, 600);
    vCanvasForward[i - 1]->cd()->SetTopMargin(0.08);
    vCanvasForward[i - 1]->cd()->SetBottomMargin(0.11);
    vCanvasForward[i - 1]->cd()->SetLeftMargin(0.12);
    vCanvasForward[i - 1]->cd()->SetRightMargin(0.16);
  }

  for (unsigned int ring = 1; ring <= n_rings; ring++) {
    h_fpix_occ[ring - 1]->Scale((100. / totalLumi_));
    SiPixelPI::makeNicePlotStyle(h_fpix_occ[ring - 1]);
    h_fpix_occ[ring - 1]->GetYaxis()->SetTitleOffset(1.2);
    h_fpix_occ[ring - 1]->GetZaxis()->SetTitleOffset(1.3);
    h_fpix_occ[ring - 1]->GetZaxis()->SetTitleSize(0.042);
    h_fpix_occ[ring - 1]->GetZaxis()->CenterTitle();

    canvasF.cd(ring)->Modified();
    SiPixelPI::dress_occup_plot(canvasF, h_fpix_occ[ring - 1], 0, ring, 1, true, true, true);
    if (useLogScale_) {
      canvasF.cd(ring)->SetLogz();
    }
    if (addLumiInfo_) {
      TPad* current_pad = static_cast<TPad*>(canvasF.cd(ring));
      CMS_lumi(current_pad, totalLumi_);
    }

    vCanvasForward[ring - 1]->cd()->Modified();
    SiPixelPI::dress_occup_plot(*vCanvasForward[ring - 1], h_fpix_occ[ring - 1], 0, ring, 1, true, true, true);

    if (useLogScale_) {
      vCanvasForward[ring - 1]->cd()->SetLogz();
    }
    if (addLumiInfo_) {
      TPad* current_pad = static_cast<TPad*>(vCanvasForward[ring - 1]->cd());
      CMS_lumi(current_pad, totalLumi_);
    }
  }

  for (unsigned int ring = 1; ring <= n_rings; ring++) {
    vCanvasForward[ring - 1]->SaveAs(("Forward_R" + std::to_string(ring) + "_" + analyzedTag_ + ".png").c_str());
    vCanvasForward[ring - 1]->SaveAs(("Forward_R" + std::to_string(ring) + "_" + analyzedTag_ + ".pdf").c_str());
  }

  canvasF.SaveAs(("SummaryForward_" + analyzedTag_ + ".png").c_str());
  canvasF.SaveAs(("SummaryForward_" + analyzedTag_ + ".pdf").c_str());
}

// ------------ purely graphics method to embellish plots  ------------
void SiPixelQualityPlotter::CMS_lumi(TPad* pad, float lumi) {
  auto ltx = TLatex();
  ltx.SetTextColor(1);
  ltx.SetTextSize(0.045);
  ltx.SetTextAlign(11);
  char str[200];
  sprintf(str, "#font[42]{L = %.3f fb^{-1}}", lumi);
  ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.01, str);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelQualityPlotter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("useLogScale", false);
  desc.add<bool>("addLumiInfo", true);
  desc.add<std::string>("analyzedTag", "");
  desc.addUntracked<unsigned int>("maxRun", 999999);
  desc.addUntracked<edm::InputTag>("lumiInputTag", edm::InputTag(""));
  descriptions.add("siPixelQualityPlotter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelQualityPlotter);
