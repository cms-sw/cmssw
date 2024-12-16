#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// the data format of the condition to be inspected
#include "CondFormats/PhysicsToolsObjects/interface/DeDxCalibration.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>

// include ROOT
#include "TCanvas.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TPad.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TStyle.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
     DeDxCalibration Payload Inspector of 1 IOV 
  *************************************************/
  class DeDxCalibrationTest : public Histogram1D<DeDxCalibration, SINGLE_IOV> {
  public:
    DeDxCalibrationTest()
        : Histogram1D<DeDxCalibration, SINGLE_IOV>("Test DeDxCalibration", "Test DeDxCalibration", 1, 0.0, 1.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<DeDxCalibration> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          const auto& thresholds = payload->thr();
          const auto& alphas = payload->alpha();
          const auto& sigmas = payload->sigma();

          assert(thresholds.size() == alphas.size());
          assert(alphas.size() == sigmas.size());

          for (unsigned int i = 0; i < thresholds.size(); i++) {
            std::cout << "threshold:" << thresholds[i] << " alpha: " << alphas[i] << " sigma: " << sigmas[i]
                      << std::endl;
          }
        }
      }
      return true;
    }
  };

  // Inspector to show the values of thresholds, alphas, sigmas, and gains in DeDxCalibration
  class DeDxCalibrationInspector : public PlotImage<DeDxCalibration, SINGLE_IOV> {
  public:
    DeDxCalibrationInspector() : PlotImage<DeDxCalibration, SINGLE_IOV>("DeDxCalibration Inspector") {}

    bool fill() override {
      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<DeDxCalibration> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        const std::vector<double>& thr = payload->thr();
        const std::vector<double>& alpha = payload->alpha();
        const std::vector<double>& sigma = payload->sigma();

        // Ensure that thr, alpha, sigma have the same size
        assert(thr.size() == alpha.size());
        assert(alpha.size() == sigma.size());

        // Create a 2D histogram
        int nBinsX = 3;           // For thr, alpha, sigma, gain
        int nBinsY = thr.size();  // Number of elements in the vectors

        TH2D h2("h2", "DeDxCalibration Values;Variable Type;Value", nBinsX, 0, nBinsX, nBinsY, 0, nBinsY);

        // Label the x-axis with the variable names
        h2.GetXaxis()->SetBinLabel(1, "Threshold");
        h2.GetXaxis()->SetBinLabel(2, "#alpha");
        h2.GetXaxis()->SetBinLabel(3, "#sigma");

        // Fill the histogram
        for (size_t i = 0; i < thr.size(); ++i) {
          h2.Fill(0.5, i, thr[i]);
          h2.Fill(1.5, i, alpha[i]);
          h2.Fill(2.5, i, sigma[i]);
        }

        // Draw the histogram on a canvas
        TCanvas canvas("Canvas", "DeDxCalibration Values", 1200, 800);
        canvas.cd();
        h2.Draw("COLZ TEXT");

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());

        return true;
      } else {
        return false;
      }
    }
  };

  class DeDxCalibrationPlot : public cond::payloadInspector::PlotImage<DeDxCalibration, SINGLE_IOV> {
  public:
    DeDxCalibrationPlot()
        : cond::payloadInspector::PlotImage<DeDxCalibration, SINGLE_IOV>(
              "DeDxCalibration Thresholds, Alphas, Sigmas, and Gains") {}

    bool fill() override {
      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<DeDxCalibration> payload = fetchPayload(std::get<1>(iov));

      if (!payload) {
        return false;
      }

      // Prepare canvas
      TCanvas canvas("DeDxCalibration", "DeDxCalibration", 1200, 800);
      canvas.Divide(2, 2);

      // Extract data
      const auto& thresholds = payload->thr();
      const auto& alphas = payload->alpha();
      const auto& sigmas = payload->sigma();
      const auto& gains = payload->gain();

      // 1. Plot thresholds
      canvas.cd(1);
      auto h_thr = new TH1F("Thresholds", "Thresholds;Index;Value", thresholds.size(), 0, thresholds.size());
      for (size_t i = 0; i < thresholds.size(); ++i) {
        h_thr->SetBinContent(i + 1, thresholds[i]);
      }
      h_thr->SetFillColor(kBlue);
      h_thr->Draw();

      // 2. Plot alphas
      canvas.cd(2);
      auto h_alpha = new TH1F("Alphas", "Alphas;Index;Value", alphas.size(), 0, alphas.size());
      for (size_t i = 0; i < alphas.size(); ++i) {
        h_alpha->SetBinContent(i + 1, alphas[i]);
      }
      h_alpha->SetFillColor(kGreen);
      h_alpha->Draw();

      // 3. Plot sigmas
      canvas.cd(3);
      auto h_sigma = new TH1F("Sigmas", "Sigmas;Index;Value", sigmas.size(), 0, sigmas.size());
      for (size_t i = 0; i < sigmas.size(); ++i) {
        h_sigma->SetBinContent(i + 1, sigmas[i]);
      }
      h_sigma->SetFillColor(kRed);
      h_sigma->Draw();

      // 4. Plot aggregated gain values
      canvas.cd(4);
      const int numBins = 100;  // Set number of bins for aggregated gain
      auto h_gain = new TH1F("Gains", "Aggregated Gains;Gain Range;Count", numBins, 0, 1.0);  // Adjust range if needed
      for (const auto& [chip, gain] : gains) {
        h_gain->Fill(gain);
      }
      h_gain->SetFillColor(kYellow);
      h_gain->Draw();

      // Save the canvas to a file
      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(DeDxCalibration) {
  PAYLOAD_INSPECTOR_CLASS(DeDxCalibrationTest);
  PAYLOAD_INSPECTOR_CLASS(DeDxCalibrationInspector);
  PAYLOAD_INSPECTOR_CLASS(DeDxCalibrationPlot);
}
