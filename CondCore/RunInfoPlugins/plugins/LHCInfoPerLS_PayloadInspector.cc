#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "utils.h"  // local
#include <sstream>
#include <memory>
#include "TLatex.h"
#include "TCanvas.h"

namespace {

  class LHCInfoPerLS_Display
      : public cond::payloadInspector::PlotImage<LHCInfoPerLS, cond::payloadInspector::SINGLE_IOV> {
  public:
    LHCInfoPerLS_Display()
        : cond::payloadInspector::PlotImage<LHCInfoPerLS, cond::payloadInspector::SINGLE_IOV>(
              "LHCInfoPerLS Inspector - Display") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<LHCInfoPerLS> payload = fetchPayload(std::get<1>(iov));
      if (!payload) {
        return false;
      }

      // Prepare label-value pairs for printing
      std::vector<std::pair<std::string, std::string>> items;
      items.push_back({"LS: ", std::to_string(payload->lumiSection())});
      items.push_back({"crossingAngleX: ", std::to_string(payload->crossingAngleX())});
      items.push_back({"crossingAngleY: ", std::to_string(payload->crossingAngleY())});
      items.push_back({"betaStarX: ", std::to_string(payload->betaStarX())});
      items.push_back({"betaStarY: ", std::to_string(payload->betaStarY())});
      items.push_back({"runNumber: ", std::to_string(payload->runNumber())});
      // Add more fields as needed

      TCanvas canvas("c", "c", 800, 600);
      canvas.cd();
      TLatex latex;
      latex.SetTextSize(0.03);

      float startY = 0.95;
      float stepY = 0.06;

      // Print the title
      latex.SetTextColor(kBlack);
      latex.DrawLatexNDC(0.05, startY, "LHCInfoPerLS Inspector");

      // Leave a blank line under the title
      std::size_t offset = 2;

      // Print each item: label in black, value in red
      for (std::size_t i = 0; i < items.size(); ++i) {
        float y = startY - (i + offset) * stepY;
        latex.SetTextColor(kBlack);
        latex.DrawLatexNDC(0.05, y, items[i].first.c_str());
        latex.SetTextColor(kRed);
        latex.DrawLatexNDC(0.30, y, items[i].second.c_str());
      }

      // info
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(12);
      t1.SetTextSize(0.04);
      t1.DrawLatex(0.1, 0.18, "LHCInfo parameters:");
      t1.DrawLatex(0.1, 0.15, "payload:");

      auto runLS = lhcInfo::unpack(std::get<0>(iov));

      t1.SetTextFont(42);
      t1.SetTextColor(4);
      t1.DrawLatex(0.37,
                   0.182,
                   Form("IOV %s (#color[2]{%s , %s})",
                        std::to_string(+std::get<0>(iov)).c_str(),
                        std::to_string(runLS.first).c_str(),
                        std::to_string(runLS.second).c_str()));
      t1.DrawLatex(0.21, 0.152, Form(" %s", (std::get<1>(iov)).c_str()));

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(LHCInfoPerLS) { PAYLOAD_INSPECTOR_CLASS(LHCInfoPerLS_Display); }
