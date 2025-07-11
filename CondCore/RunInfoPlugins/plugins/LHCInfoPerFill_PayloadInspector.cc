#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "utils.h"  // local
#include <sstream>
#include <memory>
#include "TLatex.h"
#include "TCanvas.h"

namespace {

  class LHCInfoPerFill_Display
      : public cond::payloadInspector::PlotImage<LHCInfoPerFill, cond::payloadInspector::SINGLE_IOV> {
  public:
    LHCInfoPerFill_Display()
        : cond::payloadInspector::PlotImage<LHCInfoPerFill, cond::payloadInspector::SINGLE_IOV>(
              "LHCInfoPerFill Inspector - Display") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<LHCInfoPerFill> payload = fetchPayload(std::get<1>(iov));
      if (!payload) {
        return false;
      }

      std::vector<std::pair<std::string, std::string>> items;
      items.push_back({"Fill Number: ", std::to_string(payload->fillNumber())});
      items.push_back({"Beam Energy: ", std::to_string(payload->energy())});
      items.push_back({"Deliv Lumi: ", std::to_string(payload->delivLumi())});
      items.push_back({"Rec Lumi: ", std::to_string(payload->recLumi())});
      items.push_back({"Inst Lumi: ", std::to_string(payload->instLumi())});
      items.push_back({"Injection Scheme: ", payload->injectionScheme()});
      items.push_back({"Colliding Bunches: ", std::to_string(payload->collidingBunches())});
      items.push_back({"Target Bunches: ", std::to_string(payload->targetBunches())});

      TCanvas canvas("c", "c", 800, 600);
      canvas.cd();
      TLatex latex;
      latex.SetTextSize(0.03);

      float startY = 0.95;
      float stepY = 0.06;

      // Print the title
      latex.SetTextColor(kBlack);
      latex.DrawLatexNDC(0.05, startY, "LHCInfoPerFill Inspector");

      // Print the fields with values in red
      for (std::size_t i = 0; i < items.size(); ++i) {
        float y = startY - (i + 2) * stepY;  // +2 to skip title and add a blank line
        // Print key in black
        latex.SetTextColor(kBlack);
        latex.DrawLatexNDC(0.05, y, items[i].first.c_str());
        // Print value in red, offset to the right (adjust offset as needed)
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

PAYLOAD_INSPECTOR_MODULE(LHCInfoPerFill) { PAYLOAD_INSPECTOR_CLASS(LHCInfoPerFill_Display); }
