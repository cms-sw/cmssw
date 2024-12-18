/*!
  \file L1TMuonGlobalParams_PayloadInspector
  \Payload Inspector Plugin for L1TMuonGlobalParams payloads
  \author Y. Chao
  \version $Revision: 1.0 $
  \date $Date: 2024/05/15 12:00:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"

#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParams_PUBLIC.h"

#include <fmt/format.h>

// include ROOT
#include "TH1F.h"
#include "TLine.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLatex.h"

namespace {

  using namespace cond::payloadInspector;

  class L1TMuonGlobalParamsInputBits : public PlotImage<L1TMuonGlobalParams, SINGLE_IOV> {
  public:
    L1TMuonGlobalParamsInputBits() : PlotImage<L1TMuonGlobalParams, SINGLE_IOV>("L1TMuonGlobalParams plot") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::string IOVsince = std::to_string(std::get<0>(iov));
      std::shared_ptr<L1TMuonGlobalParams> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        /// Create a canvas
        edm::LogInfo("L1TMG") << "absIsoCheckMemLUTPath: " << payload->absIsoCheckMemLUTPath();
        TCanvas canvas("L1TMuonGlobal", "L1TMuonGlobal", 800, 600);

        L1TMuonGlobalParams l1tmg = (L1TMuonGlobalParams)*payload;
        L1TMuonGlobalParamsHelper l1tmgph(l1tmg);

        canvas.cd();
        canvas.Update();

        TLatex tl;
        // Draw the columns titles
        tl.SetTextAlign(12);
        tl.SetTextSize(0.03);

        TH1F input1("InputsToDisable", "", 72, 0, 72);
        TH1F input2("MaskedInputs", "", 72, 0, 72);

        TLegend leg(0.60, 0.65, 0.85, 0.85);

        TLine lzero(0.0, 2.0, 72., 2.0);
        lzero.SetLineWidth(1);
        lzero.SetLineColor(1);
        lzero.SetLineStyle(2);

        leg.AddEntry(&input2, "MaskedInputs", "l");
        leg.AddEntry(&input1, "InputsToDisable", "l");
        leg.SetLineColor(0);
        leg.SetFillColor(0);

        input1.SetStats(false);
        input1.SetMaximum(5);
        input1.SetXTitle("InputBits");
        input1.SetYTitle("Bit value");
        input1.SetLabelOffset(0.9, "Y");
        input1.SetLineWidth(3);
        input1.SetLineColor(9);
        input2.SetLineWidth(3);
        input1.SetLineColor(8);

        for (size_t idx = 1; idx <= 72; idx++) {
          input1.SetBinContent(idx, l1tmgph.inputsToDisable()[idx] + 0.01);
          input2.SetBinContent(idx, l1tmgph.maskedInputs()[idx] + 2.01);
        }

        canvas.cd();
        input1.Draw("");
        input2.Draw("same");
        leg.Draw();
        lzero.Draw();

        auto const label_fw =
            fmt::format("fwVersion: {}, bx Min, Max: {}, {}", l1tmgph.fwVersion(), payload->bxMin(), payload->bxMax());
        auto const label_tag = fmt::format("{}, iov: {}", tag.name, IOVsince);
        tl.DrawLatexNDC(0.12, 0.85, label_fw.c_str());
        tl.DrawLatexNDC(0.10, 0.92, label_tag.c_str());
        tl.DrawLatexNDC(0.07, 0.59, "1");
        tl.DrawLatexNDC(0.07, 0.27, "1");

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }  // payload
      return true;
    }  // fill
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(L1TMuonGlobalParams) { PAYLOAD_INSPECTOR_CLASS(L1TMuonGlobalParamsInputBits); }
