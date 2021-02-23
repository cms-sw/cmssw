/*!
  \file SiPixelQualityProbabilities_PayloadInspector
  \Payload Inspector Plugin for SiPixelQualityProbabilities
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2019/10/22 19:16:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TGaxis.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
  1d histogram of SiPixelQualityProbabilities of 1 IOV 
  *************************************************/

  class SiPixelQualityProbabilitiesScenariosCount : public PlotImage<SiPixelQualityProbabilities, SINGLE_IOV> {
  public:
    SiPixelQualityProbabilitiesScenariosCount()
        : PlotImage<SiPixelQualityProbabilities, SINGLE_IOV>("SiPixelQualityProbabilities scenarios count") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelQualityProbabilities> payload = fetchPayload(std::get<1>(iov));
      auto PUbins = payload->getPileUpBins();
      auto span = PUbins.back() - PUbins.front();

      TGaxis::SetMaxDigits(3);

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>("Count",
                                       "SiPixelQualityProbablities Scenarios count;PU bin;n. of scenarios",
                                       span,
                                       PUbins.front(),
                                       PUbins.back());
      h1->SetStats(false);

      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.12);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      for (const auto &bin : PUbins) {
        h1->SetBinContent(bin + 1, payload->nelements(bin));
      }

      h1->SetTitle("");
      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetFillColor(kRed);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("bar2");

      SiPixelPI::makeNicePlotStyle(h1.get());

      canvas.Update();

      TLegend legend = TLegend(0.40, 0.88, 0.95, 0.94);
      legend.SetHeader(("Payload hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      //legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.1,
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixelQualityProbabilities IOV:" + std::to_string(std::get<0>(iov))).c_str());

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQualityProbabilities) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityProbabilitiesScenariosCount);
}
