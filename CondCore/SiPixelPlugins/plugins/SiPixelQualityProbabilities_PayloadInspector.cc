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
#include <fmt/printf.h>

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
  1d histogram of n. scenarios per PU bin in 1 IOV of SiPixelQualityProbabilities
  *************************************************/

  class SiPixelQualityProbabilitiesScenariosCount : public PlotImage<SiPixelQualityProbabilities, SINGLE_IOV> {
  public:
    SiPixelQualityProbabilitiesScenariosCount()
        : PlotImage<SiPixelQualityProbabilities, SINGLE_IOV>("SiPixelQualityProbabilities scenarios count") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiPixelQualityProbabilities> payload = fetchPayload(std::get<1>(iov));
      auto PUbins = payload->getPileUpBins();
      auto span = PUbins.back() - PUbins.front();

      TGaxis::SetMaxDigits(3);

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>("Count",
                                       "SiPixelQualityProbablities Scenarios count;PU bin;n. of scenarios per PU bin",
                                       span,
                                       PUbins.front(),
                                       PUbins.back());
      h1->SetStats(false);

      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.12);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      for (const auto& bin : PUbins) {
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

      TLegend legend = TLegend(0.40, 0.88, 0.94, 0.93);
      legend.SetHeader(("Payload hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");    // option "C" allows to center the header
      legend.SetBorderSize(0);  // Set the border size to zero to remove visible borders
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.037);
      ltx.SetTextAlign(11);

      const auto& headerText =
          fmt::sprintf("#color[4]{%s},IOV: #color[4]{%s}", tagname, std::to_string(std::get<0>(iov)));

      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.1, 1 - gPad->GetTopMargin() + 0.01, headerText.c_str());

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
  Probability density per PU bin of 1 IOV of SiPixelQualityProbabilities
  *************************************************/

  class SiPixelQualityProbabilityDensityPerPUbin : public PlotImage<SiPixelQualityProbabilities, SINGLE_IOV> {
  public:
    SiPixelQualityProbabilityDensityPerPUbin() : PlotImage<SiPixelQualityProbabilities, SINGLE_IOV>("") {
      PlotBase::addInputParam("PU bin");
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiPixelQualityProbabilities> payload = fetchPayload(std::get<1>(iov));
      auto PUbins = payload->getPileUpBins();

      // initialize the PUbin
      unsigned int PUbin(0);
      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("PU bin");
      if (ip != paramValues.end() && !ip->second.empty()) {
        PUbin = std::stoul(ip->second);
      } else {
        edm::LogWarning("SiPixelQualityProbabilityDensityPerPUbin")
            << "\n WARNING!!!! \n The needed parameter 'PU bin' has not been passed. Will use all PU bins! \n";
        PUbin = k_ALLPUBINS;
      }

      // graphics
      TGaxis::SetMaxDigits(3);

      SiPixelQualityProbabilities::probabilityVec probVec;
      if (PUbin != k_ALLPUBINS) {
        probVec = payload->getProbabilities(PUbin);
      } else {
        if (PUbins.front() == 0) {
          // if a PU bin = 0 exist the PU-averaged is in bin=0
          probVec = payload->getProbabilities(0);
        } else {
          // we need to build the PDF by hand
          // create a list of the probabilities for all the PU bins
          std::vector<SiPixelQualityProbabilities::probabilityVec> listOfProbabilityVec;
          for (unsigned int bin = PUbins.front(); bin <= PUbins.back(); bin++) {
            const auto& probsForBin = payload->getProbabilities(bin);
            listOfProbabilityVec.push_back(probsForBin);
          }

          // Map to store string and pair of floats (sum and count)
          std::map<std::string, std::pair<float, int>> stringFloatMap;

          // Loop through the list of probabilityVec elements
          for (const auto& vec : listOfProbabilityVec) {
            // For each pair in the current vector
            for (const auto& pair : vec) {
              const std::string& currentScen = pair.first;
              const float& currentProb = pair.second;

              // Check if the string exists in the map
              auto it = stringFloatMap.find(currentScen);
              if (it != stringFloatMap.end()) {
                // If the scenario already exists, update the probability sum and count
                it->second.first += currentProb;
                it->second.second++;
              } else {
                // If the string doesn't exist, add it to the map
                stringFloatMap[currentScen] = {currentProb, 1};  // Initialize sum and count
              }
            }
          }

          // Calculate the average and populate the new probabilityVec from the map
          for (const auto& pair : stringFloatMap) {
            float average = pair.second.first / pair.second.second;  // Calculate average
            probVec.emplace_back(pair.first, average);
          }
        }  // if the first PU bin is not 0
      }  // if we're asking for all the PU bins

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>("SiPixelQuality PDF",
                                       "probability density vs scenario; scenario serial ID number; probability",
                                       probVec.size(),
                                       -0.5,
                                       probVec.size() - 0.5);
      h1->SetStats(false);

      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.12);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.09);
      canvas.Modified();

      unsigned int count{0};
      for (const auto& [name, prob] : probVec) {
        h1->SetBinContent(count, prob);
        count++;
      }

      h1->SetTitle("");
      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetFillColor(kRed);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("bar2");

      SiPixelPI::makeNicePlotStyle(h1.get());

      canvas.Update();

      TLegend legend = TLegend(0.39, 0.88, 0.89, 0.93);
      std::string puBinString = (PUbin == k_ALLPUBINS) ? "PU bin: #bf{all}" : fmt::sprintf("PU bin: #bf{%u}", PUbin);
      legend.SetHeader(("#splitline{Payload hash: #bf{" + (std::get<1>(iov)) + "}}{" + puBinString + "}").c_str(),
                       "C");    // option "C" allows to center the header
      legend.SetBorderSize(0);  // Set the border size to zero to remove visible borders
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.03);
      ltx.SetTextAlign(11);

      const auto& headerText =
          fmt::sprintf("#color[4]{%s}, IOV: #color[4]{%s}", tagname, std::to_string(std::get<0>(iov)));

      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.1, 1 - gPad->GetTopMargin() + 0.01, headerText.c_str());

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    static constexpr unsigned int k_ALLPUBINS = 9999;
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQualityProbabilities) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityProbabilitiesScenariosCount);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityProbabilityDensityPerPUbin);
}
