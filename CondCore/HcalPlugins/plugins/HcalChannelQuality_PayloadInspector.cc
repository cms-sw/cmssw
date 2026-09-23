#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

  class HcalChannelStatusContainer : public HcalObjRepresent::HcalDataContainer<HcalChannelQuality, HcalChannelStatus> {
  public:
    HcalChannelStatusContainer(std::shared_ptr<HcalChannelQuality> payload, unsigned int run)
        : HcalObjRepresent::HcalDataContainer<HcalChannelQuality, HcalChannelStatus>(payload, run) {}
    float getValue(const HcalChannelStatus* chan) override { return chan->getValue() / 32770; }
  };

  /******************************************
     Detector map of HCAL ChannelStatus for 1 IOV
  ******************************************/
  class HcalChannelQualityPlot : public cond::payloadInspector::PlotImage<HcalChannelQuality> {
  public:
    HcalChannelQualityPlot() : cond::payloadInspector::PlotImage<HcalChannelQuality>("HCAL ChannelStatus - map") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [time, hash] = iovs.front();
      std::shared_ptr<HcalChannelQuality> payload = fetchPayload(hash);
      if (!payload)
        return false;
      auto container = std::make_unique<HcalChannelStatusContainer>(payload, time);
      std::unique_ptr<TCanvas> canvas(container->getCanvasAll());
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  /**********************************************************
     Detector map of HCAL ChannelStatus difference between 2 IOVs
     (arithmetic difference of the normalized status value)
  **********************************************************/
  class HcalChannelQualityChange : public cond::payloadInspector::PlotImage<HcalChannelQuality> {
  public:
    HcalChannelQualityChange()
        : cond::payloadInspector::PlotImage<HcalChannelQuality>("HCAL ChannelStatus difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [t1, h1] = iovs.front();
      auto [t2, h2] = iovs.back();

      std::shared_ptr<HcalChannelQuality> payload1 = fetchPayload(h1);
      std::shared_ptr<HcalChannelQuality> payload2 = fetchPayload(h2);
      if (!payload1 || !payload2)
        return false;

      auto container1 = std::make_unique<HcalChannelStatusContainer>(payload1, t1);
      auto container2 = std::make_unique<HcalChannelStatusContainer>(payload2, t2);
      container2->Subtract(container1.get());
      std::unique_ptr<TCanvas> canvas(container2->getCanvasAll());
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  /**********************************************************
     Categorical map of channel-status changes between 2 IOVs:
     which channels flipped good<->bad, split by subdetector/depth,
     laid out the same way as HcalObjRepresent::getCanvasAll().

     "Bad" is defined as any of the DQM-derived quality bits being set:
     HcalCellOff, HcalCellDead, HcalCellHot, HcalCellStabErr, HcalCellTimErr,
     HcalBadLaserSignal. 
  **********************************************************/
  class HcalChannelQualityStatusChangeMap : public cond::payloadInspector::PlotImage<HcalChannelQuality> {
  public:
    HcalChannelQualityStatusChangeMap()
        : cond::payloadInspector::PlotImage<HcalChannelQuality>("HCAL ChannelStatus - status change map") {
      setSingleIov(false);
    }

    // category codes -- kept nonzero so a real "unchanged good" channel
    // is still distinguishable from an empty/no-channel bin (which ROOT
    // leaves at 0).
    enum Category { NewlyBad = -2, StillBad = -1, StillGood = 1, NewlyGood = 2 };

    static bool isBad(const HcalChannelStatus& chan) {
      static const std::vector<unsigned int> badBits = {
          HcalChannelStatus::HcalCellOff,
          HcalChannelStatus::HcalCellDead,
          HcalChannelStatus::HcalCellHot,
          HcalChannelStatus::HcalCellStabErr,
          HcalChannelStatus::HcalCellTimErr,
          HcalChannelStatus::HcalBadLaserSignal,
      };
      for (unsigned int bit : badBits) {
        if (chan.isBitSet(bit))
          return true;
      }
      return false;
    }

    using Coord = std::tuple<int, int, int>;  // (depth, ieta, iphi)
    using DepthKey = std::pair<std::string, int>;
    using StatusMap = std::map<DepthKey, std::map<Coord, bool>>;

    static StatusMap buildStatusMap(const std::vector<std::pair<std::string, std::vector<HcalChannelStatus>>>& conts) {
      StatusMap out;
      for (const auto& cont : conts) {
        const std::string& subDetName = cont.first;
        if (subDetName.empty() || subDetName[0] != 'H')
          continue;
        for (const auto& item : cont.second) {
          HcalDetId id(item.rawId());
          int depth = id.depth();
          if (depth == 0)
            continue;
          out[std::make_pair(subDetName, depth)][std::make_tuple(depth, id.ieta(), id.iphi())] = isBad(item);
        }
      }
      return out;
    }

    // Mirrors HcalObjRepresent::HcalDataContainer::FillCanv's pad placement,
    // but colors by category instead of a continuous colz range.
    void drawCategoryPad(TCanvas* canvas,
                         const std::string& subDetName,
                         int startDepth,
                         int startCanv,
                         const std::map<std::pair<std::string, int>, std::unique_ptr<TH2F>>& hists,
                         int maxDepth) {
      for (int depth = startDepth; depth <= maxDepth; ++depth) {
        auto it = hists.find(std::make_pair(subDetName, depth));
        if (it == hists.end())
          return;

        int padNum = depth + startCanv - 1;
        if (subDetName == "HO")
          padNum -= 3;

        canvas->cd(padNum);
        canvas->GetPad(padNum)->SetGridx(1);
        canvas->GetPad(padNum)->SetGridy(1);
        canvas->GetPad(padNum)->SetRightMargin(0.13);

        TH2F* h = it->second.get();
        h->GetZaxis()->SetRangeUser(-2.5, 2.5);
        h->SetContour(4);
        h->GetXaxis()->SetTitle("ieta");
        h->GetYaxis()->SetTitle("iphi");
        h->GetXaxis()->CenterTitle();
        h->GetYaxis()->CenterTitle();
        h->Draw("col");

        TLatex label;
        label.SetNDC();
        label.SetTextAlign(22);
        label.SetTextSize(0.06);
        label.DrawLatex(0.5, 0.95, (subDetName + " depth " + std::to_string(depth)).c_str());
      }
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [t1, h1] = iovs.front();
      auto [t2, h2] = iovs.back();

      std::shared_ptr<HcalChannelQuality> payload1 = fetchPayload(h1);
      std::shared_ptr<HcalChannelQuality> payload2 = fetchPayload(h2);
      if (!payload1 || !payload2)
        return false;

      // Reuse the container only to determine TopoMode / per-subdet depth
      // counts via its public accessors -- GetDepths() must run once to
      // populate them.
      auto refContainer = std::make_unique<HcalChannelStatusContainer>(payload2, t2);
      refContainer->GetDepths();
      std::string topoMode = refContainer->GetTopoMode();
      auto subDetDepths = refContainer->GetSubDetDepths();

      StatusMap firstStatus = buildStatusMap(payload1->getAllContainers());
      StatusMap lastStatus = buildStatusMap(payload2->getAllContainers());

      // custom 4-color discrete palette: red / orange / light green / dark green
      TColor::InitializeColors();
      Int_t colors[4] = {kRed + 1, kOrange + 1, kGreen - 9, kGreen + 2};
      gStyle->SetPalette(4, colors);
      gStyle->SetOptStat(0);
      gStyle->SetNumberContours(4);

      std::map<std::pair<std::string, int>, std::unique_ptr<TH2F>> categoryHists;

      for (const auto& [key, lastCoords] : lastStatus) {
        auto firstIt = firstStatus.find(key);
        if (firstIt == firstStatus.end())
          continue;
        const auto& firstCoords = firstIt->second;

        auto hist = std::make_unique<TH2F>(
            ("change_" + key.first + "_d" + std::to_string(key.second)).c_str(), "", 84, -42.5, 41.5, 72, 0.5, 72.5);

        for (const auto& [coord, badLast] : lastCoords) {
          auto firstCoordIt = firstCoords.find(coord);
          if (firstCoordIt == firstCoords.end())
            continue;  // channel not present in the earlier IOV -- skip
          bool badFirst = firstCoordIt->second;

          int category;
          if (!badFirst && !badLast)
            category = StillGood;
          else if (badFirst && badLast)
            category = StillBad;
          else if (!badFirst && badLast)
            category = NewlyBad;
          else
            category = NewlyGood;

          int ieta = std::get<1>(coord);
          int iphi = std::get<2>(coord);
          hist->Fill(ieta, iphi, category);
        }

        categoryHists[key] = std::move(hist);
      }

      if (categoryHists.empty())
        return false;

      int rows = (topoMode == "2015/2016") ? 3 : 6;
      TCanvas canvas("HChange", "HChange", 1680, (topoMode == "2015/2016") ? 1680 : 2500);
      canvas.Divide(3, rows, 0.02, 0.01);

      drawCategoryPad(&canvas, "HB", 1, 1, categoryHists, subDetDepths["HB"]);
      drawCategoryPad(&canvas, "HO", 4, 3, categoryHists, 4);
      drawCategoryPad(&canvas, "HF", 1, 4, categoryHists, subDetDepths["HF"]);
      drawCategoryPad(&canvas, "HE", 1, (topoMode == "2015/2016") ? 7 : 10, categoryHists, subDetDepths["HE"]);

      TLatex legend;
      legend.SetNDC();
      legend.SetTextSize(0.02);
      legend.DrawLatex(
          0.01, 0.005, "Red = newly bad   Orange = still bad   Light green = still good   Dark green = newly good");

      canvas.SaveAs(this->m_imageFileName.c_str());
      return true;
    }  // fill method
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalChannelQuality) {
  PAYLOAD_INSPECTOR_CLASS(HcalChannelQualityPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalChannelQualityChange);
  PAYLOAD_INSPECTOR_CLASS(HcalChannelQualityStatusChangeMap);
}
