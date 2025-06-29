#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>
#include <memory>
#include "TLatex.h"
#include "TCanvas.h"

namespace {

  class LHCInfoPerLS_Display : public cond::payloadInspector::PlotImage<LHCInfoPerLS> {
  public:
    LHCInfoPerLS_Display() : cond::payloadInspector::PlotImage<LHCInfoPerLS>("LHCInfoPerLS Inspector - Display") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<LHCInfoPerLS> payload = fetchPayload(std::get<1>(iov));
      if (!payload) {
        return false;
      }
      // Prepare lines for printing
      std::vector<std::string> lines;
      lines.push_back("LHCInfoPerLS Inspector");
      lines.push_back("");
      lines.push_back("LS: " + std::to_string(payload->lumiSection()));
      lines.push_back("crossingAngleX: " + std::to_string(payload->crossingAngleX()));
      lines.push_back("crossingAngleY: " + std::to_string(payload->crossingAngleY()));
      lines.push_back("betaStarX: " + std::to_string(payload->betaStarX()));
      lines.push_back("betaStarY: " + std::to_string(payload->betaStarY()));
      lines.push_back("runNumber: " + std::to_string(payload->runNumber()));
      // Add more fields as needed

      TCanvas canvas("c","c",800,600);
      canvas.cd();
      TLatex latex;
      latex.SetTextSize(0.03);
      
      float startY = 0.95;
      float stepY = 0.06;
      for (std::size_t i = 0; i < lines.size(); ++i) {
	latex.DrawLatexNDC(0.05, startY - i * stepY, lines[i].c_str());
      }
      std::string fileName = "LHCInfoPerLS.png";
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(LHCInfoPerLS) { PAYLOAD_INSPECTOR_CLASS(LHCInfoPerLS_Display); }
