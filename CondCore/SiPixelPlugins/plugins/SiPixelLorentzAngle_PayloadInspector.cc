/*!
  \file SiPixelLorentzAngle_PayloadInspector
  \Payload Inspector Plugin for SiPixel Lorentz angles
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2019/06/20 10:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"

#include <memory>
#include <sstream>

// include ROOT
#include "TH2F.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace {

  /************************************************
    1d histogram of SiPixelLorentzAngle of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiPixelLorentzAngleValue : public cond::payloadInspector::Histogram1D<SiPixelLorentzAngle> {
  public:
    SiPixelLorentzAngleValue()
        : cond::payloadInspector::Histogram1D<SiPixelLorentzAngle>(
              "SiPixel LorentzAngle values", "SiPixel LorentzAngle values", 100, 0.0, 0.1) {
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>> &iovs) override {
      for (auto const &iov : iovs) {
        std::shared_ptr<SiPixelLorentzAngle> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();

          for (const auto &element : LAMap_) {
            fillWithValue(element.second);
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
    1d histogram of SiPixelLorentzAngle of 1 IOV 
  *************************************************/
  class SiPixelLorentzAngleValues : public cond::payloadInspector::PlotImage<SiPixelLorentzAngle> {
  public:
    SiPixelLorentzAngleValues() : cond::payloadInspector::PlotImage<SiPixelLorentzAngle>("SiPixelLorentzAngle") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>> &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiPixelLorentzAngle> payload = fetchPayload(std::get<1>(iov));
      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto h1 = std::unique_ptr<TH1F>(
          new TH1F("value", "SiPixel LA value;SiPixel Lorentz Angle [rad];# modules", 100, 0., 0.1));
      h1->SetStats(false);
      canvas.SetBottomMargin(0.10);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.03);
      canvas.Modified();

      for (const auto &element : LAMap_) {
        h1->Fill(element.second);
      }

      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetFillColor(kRed);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("HIST");
      h1->Draw("Psame");

      canvas.Update();

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
      legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
    1d histogram of SiPixelLorentzAngle of 1 IOV 
  *************************************************/
  class SiPixelLorentzAngleValueComparisonBase : public cond::payloadInspector::PlotImage<SiPixelLorentzAngle> {
  public:
    SiPixelLorentzAngleValueComparisonBase()
        : cond::payloadInspector::PlotImage<SiPixelLorentzAngle>("SiPixelLorentzAngle") {}
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>> &iovs) override {
      TH1F::SetDefaultSumw2(true);
      std::vector<std::tuple<cond::Time_t, cond::Hash>> sorted_iovs = iovs;
      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });
      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<SiPixelLorentzAngle> last_payload = fetchPayload(std::get<1>(lastiov));
      std::map<uint32_t, float> l_LAMap_ = last_payload->getLorentzAngles();
      std::shared_ptr<SiPixelLorentzAngle> first_payload = fetchPayload(std::get<1>(firstiov));
      std::map<uint32_t, float> f_LAMap_ = first_payload->getLorentzAngles();

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto hfirst = std::unique_ptr<TH1F>(
          new TH1F("value_first", "SiPixel LA value;SiPixel Lorentz Angle [rad];# modules", 50, 0., 0.1));
      hfirst->SetStats(false);

      auto hlast = std::unique_ptr<TH1F>(
          new TH1F("value_last", "SiPixel LA value;SiPixel Lorentz Angle [rad];# modules", 50, 0., 0.1));
      hlast->SetStats(false);

      canvas.SetBottomMargin(0.10);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.03);
      canvas.Modified();

      for (const auto &element : f_LAMap_) {
        hfirst->Fill(element.second);
      }

      for (const auto &element : l_LAMap_) {
        hlast->Fill(element.second);
      }

      auto extrema = SiPixelPI::getExtrema(hfirst.get(), hlast.get());
      hfirst->GetYaxis()->SetRangeUser(extrema.first, extrema.second * 1.10);

      hfirst->SetFillColor(kRed);
      hfirst->SetMarkerStyle(20);
      hfirst->SetMarkerSize(1);
      hfirst->Draw("HIST");
      hfirst->Draw("Psame");

      hlast->SetFillColorAlpha(kBlue, 0.20);
      hlast->SetMarkerStyle(20);
      hlast->SetMarkerSize(1);
      hlast->Draw("HISTsame");
      hlast->Draw("Psame");

      canvas.Update();

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader("SiPixel Lorentz Angle Comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(hfirst.get(), ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "FL");
      legend.AddEntry(hlast.get(), ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "FL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  class SiPixelLorentzAngleValueComparisonSingleTag : public SiPixelLorentzAngleValueComparisonBase {
  public:
    SiPixelLorentzAngleValueComparisonSingleTag() : SiPixelLorentzAngleValueComparisonBase() { setSingleIov(false); }
  };

  class SiPixelLorentzAngleValueComparisonTwoTags : public SiPixelLorentzAngleValueComparisonBase {
  public:
    SiPixelLorentzAngleValueComparisonTwoTags() : SiPixelLorentzAngleValueComparisonBase() { setTwoTags(true); }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelLorentzAngle) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValue);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValueComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValueComparisonTwoTags);
}
