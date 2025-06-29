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
      std::ostringstream ss;
      ss << "LHCInfoPerFill Inspector\n\n";
      ss << "Fill Number: " << payload->fillNumber() << "\n";
      ss << "Beam Energy: " << payload->energy() << "\n";
      ss << "Deliv Lumi: " << payload->delivLumi() << "\n";
      ss << "Rec Lumi: " << payload->recLumi() << "\n";
      ss << "Inst Lumi: " << payload->instLumi() << "\n";
      ss << "Injection Scheme: " << payload->injectionScheme() << "\n";
      ss << "Colliding Bunches: " << payload->collidingBunches() << "\n";
      ss << "Target Bunches: " << payload->targetBunches() << "\n";
      // Add more fields as needed

      std::string outText = ss.str();
      // Save as a simple image (text on white bg)
      TCanvas canvas("c", "c", 800, 600);
      TLatex latex;
      latex.SetTextSize(0.03);
      latex.DrawLatexNDC(0.05, 0.95, outText.c_str());
      std::string fileName = "LHCInfoPerFill.png";
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(LHCInfoPerFill) { PAYLOAD_INSPECTOR_CLASS(LHCInfoPerFill_Display); }
