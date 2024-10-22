#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/Luminosity/interface/LumiCorrections.h"

#include <memory>
#include <sstream>
#include <iostream>

#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"

namespace {

  /************************************************
    summary class
  *************************************************/

  class LumiCorrectionsSummary : public cond::payloadInspector::PlotImage<LumiCorrections> {
  public:
    LumiCorrectionsSummary() : cond::payloadInspector::PlotImage<LumiCorrections>("LumiCorrections Summary") {}

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      std::shared_ptr<LumiCorrections> payload = fetchPayload(std::get<1>(iov));
      auto unpacked = unpack(std::get<0>(iov));
      if (payload != nullptr) {
        TCanvas canvas(Form("LumiCorrectionsSummary per BX Run %d Lumi %d", unpacked.first, unpacked.second),
                       Form("LumiCorrectionsSummary per BX Run %d Lumi %d", unpacked.first, unpacked.second),
                       1200,
                       600);
        canvas.cd();
        const int nBX = 3564;
        std::vector<float> correctionScaleFactors_ = payload->getCorrectionsBX();
        Double_t x[nBX];
        Double_t y[nBX];
        for (int i = 0; i < nBX; i++) {
          x[i] = i;
          y[i] = correctionScaleFactors_[i];
        }
        TGraph* gr = new TGraph(nBX, x, y);
        gr->SetTitle(Form("LumiCorrectionsSummary per BX Run %d Lumi %d", unpacked.first, unpacked.second));
        gr->Draw("AP");
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

        return true;
      } else {
        return false;
      }
    }

    std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
      auto kLowMask = 0XFFFFFFFF;
      auto run = (since >> 32);
      auto lumi = (since & kLowMask);
      return std::make_pair(run, lumi);
    }
  };
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(LumiCorrections) { PAYLOAD_INSPECTOR_CLASS(LumiCorrectionsSummary); }
