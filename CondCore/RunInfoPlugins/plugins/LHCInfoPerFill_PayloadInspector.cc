#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>
#include <memory>
#include "TLatex.h"
#include "TCanvas.h"

namespace {

  class LHCInfoPerFill_Display : public cond::payloadInspector::PlotImage<LHCInfoPerFill> {
  public:
    LHCInfoPerFill_Display()
        : cond::payloadInspector::PlotImage<LHCInfoPerFill>("LHCInfoPerFill Inspector - Display") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<LHCInfoPerFill> payload = fetchPayload(std::get<1>(iov));
      if (!payload) {
        return false;
      }

      std::vector<std::string> lines;
      lines.push_back("LHCInfoPerFill Inspector");
      lines.push_back("");
      lines.push_back("Fill Number: " + std::to_string(payload->fillNumber()));
      lines.push_back("Beam Energy: " + std::to_string(payload->energy()));
      lines.push_back("Deliv Lumi: " + std::to_string(payload->delivLumi()));
      lines.push_back("Rec Lumi: " + std::to_string(payload->recLumi()));
      lines.push_back("Inst Lumi: " + std::to_string(payload->instLumi()));
      lines.push_back("Injection Scheme: " + payload->injectionScheme());
      lines.push_back("Colliding Bunches: " + std::to_string(payload->collidingBunches()));
      lines.push_back("Target Bunches: " + std::to_string(payload->targetBunches()));

      TCanvas canvas("c","c",800,600);
      canvas.cd();
      TLatex latex;
      latex.SetTextSize(0.03);
      
      float startY = 0.95;
      float stepY = 0.06;
      for (std::size_t i = 0; i < lines.size(); ++i) {
	latex.DrawLatexNDC(0.05, startY - i * stepY, lines[i].c_str());
      }
      std::string fileName = "LHCInfoPerFill.png";
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(LHCInfoPerFill) { PAYLOAD_INSPECTOR_CLASS(LHCInfoPerFill_Display); }
