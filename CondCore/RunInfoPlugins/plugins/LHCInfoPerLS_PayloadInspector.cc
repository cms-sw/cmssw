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
      std::ostringstream ss;
      ss << "LHCInfoPerLS Inspector\n\n";
      ss << "LS: " << payload->lumiSection() << "\n";
      ss << "crossingAngleX: " << payload->crossingAngleX() << "\n";
      ss << "crossingAngleY: " << payload->crossingAngleY() << "\n";
      ss << "betaStarX: " << payload->betaStarX() << "\n";
      ss << "betaStarY: " << payload->betaStarY() << "\n";
      ss << "runNumber: " << payload->runNumber() << "\n";
      // Add more fields as needed

      std::string outText = ss.str();
      // Save as a simple image (text on white bg)
      TCanvas canvas("c", "c", 800, 600);
      TLatex latex;
      latex.SetTextSize(0.03);
      latex.DrawLatexNDC(0.05, 0.95, outText.c_str());
      std::string fileName = "LHCInfoPerLS.png";
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(LHCInfoPerLS) { PAYLOAD_INSPECTOR_CLASS(LHCInfoPerLS_Display); }
