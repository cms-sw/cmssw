/*!
  \file SiStripConfObject_PayloadInspector
  \Payload Inspector Plugin for SiStrip Configuration Objects
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/04/08 18:29:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// helper function
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"

#include <memory>
#include <sstream>
#include <iostream>
#include <regex>

// include ROOT
#include "TH2F.h"
#include "TF1.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"

namespace {

  using namespace cond::payloadInspector;

  // test class
  class SiStripConfObjectTest : public Histogram1D<SiStripConfObject, SINGLE_IOV> {
  public:
    SiStripConfObjectTest()
        : Histogram1D<SiStripConfObject, SINGLE_IOV>(
              "SiStrip Configuration Object test", "SiStrip Configuration Object test", 1, 0.0, 1.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripConfObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          std::stringstream ss;
          ss << "Summary of strips configuration object:" << std::endl;

          SiStripConfObject::parMap::const_iterator it = payload->parameters.begin();
          for (; it != payload->parameters.end(); ++it) {
            ss << "parameter name = " << it->first << " value = " << it->second << std::endl;
          }

          std::cout << ss.str() << std::endl;

        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  // display class
  class SiStripConfObjectDisplay : public PlotImage<SiStripConfObject, SINGLE_IOV> {
  public:
    SiStripConfObjectDisplay() : PlotImage<SiStripConfObject, SINGLE_IOV>("Display Configuration Values") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripConfObject> payload = fetchPayload(std::get<1>(iov));

      unsigned int run = std::get<0>(iov);
      std::vector<float> y_line;

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.045);
      t1.SetTextColor(2);

      TLatex latex;
      latex.SetNDC();
      latex.SetTextSize(0.035);

      unsigned int configsize_ = payload->parameters.size();
      TLine lines[configsize_ + 1];

      auto h_Config = std::make_unique<TH1F>("ConfigParamter", ";;configuration value", configsize_, 0., configsize_);
      h_Config->SetStats(false);

      bool isShiftAndXTalk = payload->isParameter("shift_IB1Deco");

      // different canvases for for different types of conditions
      int c_width = isShiftAndXTalk ? 2000 : 1000;
      int c_height = isShiftAndXTalk ? 1000 : 800;

      TCanvas canvas("Configuration Summary", "Configuration Summary", c_width, c_height);
      canvas.cd();

      // if its for APV phase offsets
      if (!isShiftAndXTalk) {
        t1.DrawLatex(0.5, 0.96, Form("SiStrip ConfObject, IOV %i", run));
        latex.DrawLatex(0.1, 0.92, "Parameter");
        latex.DrawLatex(0.6, 0.92, "Value");
        y_line.push_back(0.92);
        latex.SetTextFont(42);
        latex.SetTextColor(kBlack);

        SiStripConfObject::parMap::const_iterator it = payload->parameters.begin();
        unsigned int count = 0;
        for (; it != payload->parameters.end(); ++it) {
          count++;
          float y_loc = 0.92 - (0.90 / configsize_) * count;
          latex.SetTextSize(std::min(0.035, 0.95 / configsize_));
          latex.DrawLatex(0.1, y_loc, (it->first).c_str());
          latex.DrawLatex(0.6, y_loc, (it->second).c_str());
          y_line.push_back(y_loc);
        }

        unsigned int iL = 0;
        for (const auto& line : y_line) {
          lines[iL] = TLine(gPad->GetUxmin(), line, gPad->GetUxmax(), line);
          lines[iL].SetLineWidth(1);
          lines[iL].SetLineStyle(9);
          lines[iL].SetLineColor(2);
          lines[iL].Draw("same");
          iL++;
        }

      }
      // if it's for shifts and cross-talk
      else {
        canvas.SetBottomMargin(0.16);
        canvas.SetLeftMargin(0.08);
        canvas.SetRightMargin(0.02);
        canvas.SetTopMargin(0.05);
        canvas.Modified();

        SiStripConfObject::parMap::const_iterator it = payload->parameters.begin();

        unsigned int count = 0;
        for (; it != payload->parameters.end(); ++it) {
          count++;

          h_Config->SetBinContent(count, std::stof(it->second));
          h_Config->GetXaxis()->SetBinLabel(count, (std::regex_replace(it->first, std::regex("_"), " ")).c_str());
        }

        SiStripPI::makeNicePlotStyle(h_Config.get());
        h_Config->GetYaxis()->SetTitleOffset(0.8);
        h_Config->GetXaxis()->SetLabelSize(0.03);
        h_Config->SetMaximum(h_Config->GetMaximum() * 1.20);
        h_Config->SetFillColorAlpha(kRed, 0.35);
        h_Config->Draw();
        h_Config->Draw("textsame");

        t1.DrawLatex(0.5, 0.96, Form("SiStrip ConfObject, IOV %i: Payload %s", run, (std::get<1>(iov)).c_str()));
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripConfObject) {
  PAYLOAD_INSPECTOR_CLASS(SiStripConfObjectTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripConfObjectDisplay);
}
