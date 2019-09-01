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
#include "FWCore/ParameterSet/interface/FileInPath.h"
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
    SiPixelLorentzAngleValues() : cond::payloadInspector::PlotImage<SiPixelLorentzAngle>("SiPixelLorentzAngle Values") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>> &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiPixelLorentzAngle> payload = fetchPayload(std::get<1>(iov));
      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto h1 = std::unique_ptr<TH1F>(new TH1F(
          "value", "SiPixel LA value;SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T];# modules", 50, 0.051, 0.15));
      h1->SetStats(false);

      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.12);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      for (const auto &element : LAMap_) {
        h1->Fill(element.second);
      }

      h1->SetTitle("");
      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetFillColor(kRed);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("HIST");
      h1->Draw("Psame");

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
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Lorentz Angle IOV:" + std::to_string(std::get<0>(iov))).c_str());

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
        : cond::payloadInspector::PlotImage<SiPixelLorentzAngle>("SiPixelLorentzAngle Values Comparison") {}
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
          new TH1F("value_first",
                   "SiPixel LA value;SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T];# modules",
                   50,
                   0.051,
                   0.15));
      hfirst->SetStats(false);

      auto hlast = std::unique_ptr<TH1F>(
          new TH1F("value_last",
                   "SiPixel LA value;SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T];# modules",
                   50,
                   0.051,
                   0.15));
      hlast->SetStats(false);

      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.12);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      for (const auto &element : f_LAMap_) {
        hfirst->Fill(element.second);
      }

      for (const auto &element : l_LAMap_) {
        hlast->Fill(element.second);
      }

      auto extrema = SiPixelPI::getExtrema(hfirst.get(), hlast.get());
      hfirst->GetYaxis()->SetRangeUser(extrema.first, extrema.second * 1.10);

      hfirst->SetTitle("");
      hfirst->SetFillColor(kRed);
      hfirst->SetMarkerStyle(kFullCircle);
      hfirst->SetMarkerSize(1.5);
      hfirst->SetMarkerColor(kRed);
      hfirst->Draw("HIST");
      hfirst->Draw("Psame");

      hlast->SetTitle("");
      hlast->SetFillColorAlpha(kBlue, 0.20);
      hlast->SetMarkerStyle(kOpenCircle);
      hlast->SetMarkerSize(1.5);
      hlast->SetMarkerColor(kBlue);
      hlast->Draw("HISTsame");
      hlast->Draw("Psame");

      SiPixelPI::makeNicePlotStyle(hfirst.get());
      SiPixelPI::makeNicePlotStyle(hlast.get());

      canvas.Update();

      TLegend legend = TLegend(0.32, 0.86, 0.95, 0.94);
      //legend.SetHeader("#font[22]{SiPixel Lorentz Angle Comparison}", "C");  // option "C" allows to center the header
      //legend.AddEntry(hfirst.get(), ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "FL");
      //legend.AddEntry(hlast.get(),  ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "FL");
      legend.AddEntry(hfirst.get(), ("payload: #color[2]{" + std::get<1>(firstiov) + "}").c_str(), "F");
      legend.AddEntry(hlast.get(), ("payload: #color[4]{" + std::get<1>(lastiov) + "}").c_str(), "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Lorentz Angle IOV: #color[2]{" + std::to_string(std::get<0>(firstiov)) +
                        "} vs IOV: #color[4]{" + std::to_string(std::get<0>(lastiov)) + "}")
                           .c_str());

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

  /************************************************
   Summary Comparison per region of SiPixelLorentzAngle between 2 IOVs
  *************************************************/
  class SiPixelLorentzAngleByRegionComparisonBase : public cond::payloadInspector::PlotImage<SiPixelLorentzAngle> {
  public:
    SiPixelLorentzAngleByRegionComparisonBase()
        : cond::payloadInspector::PlotImage<SiPixelLorentzAngle>("SiPixelLorentzAngle Comparison by Region") {}
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>> &iovs) override {
      gStyle->SetPaintTextFormat(".3f");

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

      TCanvas canvas("Comparison", "Comparison", 1600, 800);

      std::map<SiPixelPI::regions, std::shared_ptr<TH1F>> FirstLA_spectraByRegion;
      std::map<SiPixelPI::regions, std::shared_ptr<TH1F>> LastLA_spectraByRegion;
      std::shared_ptr<TH1F> summaryFirst;
      std::shared_ptr<TH1F> summaryLast;

      // book the intermediate histograms
      for (int r = SiPixelPI::BPixL1o; r != SiPixelPI::NUM_OF_REGIONS; r++) {
        SiPixelPI::regions part = static_cast<SiPixelPI::regions>(r);
        std::string s_part = SiPixelPI::getStringFromRegionEnum(part);

        FirstLA_spectraByRegion[part] = std::make_shared<TH1F>(Form("hfirstLA_%s", s_part.c_str()),
                                                               Form(";%s #mu_{H} [1/T];n. of modules", s_part.c_str()),
                                                               1000,
                                                               0.,
                                                               1000.);
        LastLA_spectraByRegion[part] = std::make_shared<TH1F>(Form("hlastLA_%s", s_part.c_str()),
                                                              Form(";%s #mu_{H} [1/T];n. of modules", s_part.c_str()),
                                                              1000,
                                                              0.,
                                                              1000.);
      }

      summaryFirst = std::make_shared<TH1F>("first Summary",
                                            "Summary for #LT tan#theta_{L}/B #GT;;average LA #LT #mu_{H} #GT [1/T]",
                                            FirstLA_spectraByRegion.size(),
                                            0,
                                            FirstLA_spectraByRegion.size());
      summaryLast = std::make_shared<TH1F>("last Summary",
                                           "Summary for #LT tan#theta_{L}/B #GT;;average LA #LT #mu_{H}  #GT [1/T]",
                                           LastLA_spectraByRegion.size(),
                                           0,
                                           LastLA_spectraByRegion.size());

      const char *path_toTopologyXML = (f_LAMap_.size() == SiPixelPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      bool isPhase0(false);
      if (f_LAMap_.size() == SiPixelPI::phase0size) {
        isPhase0 = true;
      }

      // -------------------------------------------------------------------
      // loop on the first LA Map
      // -------------------------------------------------------------------
      for (const auto &it : f_LAMap_) {
        if (DetId(it.first).det() != DetId::Tracker) {
          edm::LogWarning("SiPixelLorentzAngle_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.first << " - terminating ";
          return false;
        }

        SiPixelPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.first);
        t_info_fromXML.fillGeometryInfo(detid, f_tTopo, isPhase0);

        SiPixelPI::regions thePart = t_info_fromXML.filterThePartition();
        FirstLA_spectraByRegion[thePart]->Fill(it.second);
      }  // ends loop on the vector of error transforms

      path_toTopologyXML = (l_LAMap_.size() == SiPixelPI::phase0size)
                               ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                               : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      if (l_LAMap_.size() == SiPixelPI::phase0size) {
        isPhase0 = true;
      }

      // -------------------------------------------------------------------
      // loop on the second LA Map
      // -------------------------------------------------------------------
      for (const auto &it : l_LAMap_) {
        if (DetId(it.first).det() != DetId::Tracker) {
          edm::LogWarning("SiPixelLorentzAngle_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.first << " - terminating ";
          return false;
        }

        SiPixelPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.first);
        t_info_fromXML.fillGeometryInfo(detid, l_tTopo, isPhase0);

        SiPixelPI::regions thePart = t_info_fromXML.filterThePartition();
        LastLA_spectraByRegion[thePart]->Fill(it.second);
      }  // ends loop on the vector of error transforms

      // fill the summary plots
      int bin = 1;
      for (int r = SiPixelPI::BPixL1o; r != SiPixelPI::NUM_OF_REGIONS; r++) {
        SiPixelPI::regions part = static_cast<SiPixelPI::regions>(r);

        summaryFirst->GetXaxis()->SetBinLabel(bin, SiPixelPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float f_mean =
            FirstLA_spectraByRegion[part]->GetMean() > 10.e-6 ? FirstLA_spectraByRegion[part]->GetMean() : 10.e-6;
        summaryFirst->SetBinContent(bin, f_mean);
        //summaryFirst->SetBinError(bin,LA_spectraByRegion[hash]->GetRMS());

        summaryLast->GetXaxis()->SetBinLabel(bin, SiPixelPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float l_mean =
            LastLA_spectraByRegion[part]->GetMean() > 10.e-6 ? LastLA_spectraByRegion[part]->GetMean() : 10.e-6;
        summaryLast->SetBinContent(bin, l_mean);
        //summaryLast->SetBinError(bin,LA_spectraByRegion[hash]->GetRMS());
        bin++;
      }

      SiPixelPI::makeNicePlotStyle(summaryFirst.get());  //, kBlue);
      summaryFirst->SetMarkerColor(kRed);
      summaryFirst->GetXaxis()->LabelsOption("v");
      summaryFirst->GetXaxis()->SetLabelSize(0.05);
      summaryFirst->GetYaxis()->SetTitleOffset(0.9);

      SiPixelPI::makeNicePlotStyle(summaryLast.get());  //, kRed);
      summaryLast->SetMarkerColor(kBlue);
      summaryLast->GetYaxis()->SetTitleOffset(0.9);
      summaryLast->GetXaxis()->LabelsOption("v");
      summaryLast->GetXaxis()->SetLabelSize(0.05);

      canvas.cd()->SetGridy();

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.02);
      canvas.Modified();

      summaryFirst->SetFillColor(kRed);
      summaryLast->SetFillColor(kBlue);

      summaryFirst->SetBarWidth(0.45);
      summaryFirst->SetBarOffset(0.1);

      summaryLast->SetBarWidth(0.4);
      summaryLast->SetBarOffset(0.55);

      summaryLast->SetMarkerSize(1.5);
      summaryFirst->SetMarkerSize(1.5);

      float max = (summaryFirst->GetMaximum() > summaryLast->GetMaximum()) ? summaryFirst->GetMaximum()
                                                                           : summaryLast->GetMaximum();

      summaryFirst->GetYaxis()->SetRangeUser(0., std::max(0., max * 1.40));

      summaryFirst->Draw("bar2");
      summaryFirst->Draw("text90same");
      summaryLast->Draw("bar2,same");
      summaryLast->Draw("text60same");

      TLegend legend = TLegend(0.52, 0.80, 0.98, 0.9);
      legend.SetHeader("#mu_{H} value comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(
          summaryLast.get(),
          ("IOV: #scale[1.2]{" + std::to_string(std::get<0>(lastiov)) + "} | #color[4]{" + std::get<1>(lastiov) + "}")
              .c_str(),
          "F");
      legend.AddEntry(
          summaryFirst.get(),
          ("IOV: #scale[1.2]{" + std::to_string(std::get<0>(firstiov)) + "} | #color[2]{" + std::get<1>(firstiov) + "}")
              .c_str(),
          "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

  class SiPixelLorentzAngleByRegionComparisonSingleTag : public SiPixelLorentzAngleByRegionComparisonBase {
  public:
    SiPixelLorentzAngleByRegionComparisonSingleTag() : SiPixelLorentzAngleByRegionComparisonBase() {
      setSingleIov(false);
    }
  };

  class SiPixelLorentzAngleByRegionComparisonTwoTags : public SiPixelLorentzAngleByRegionComparisonBase {
  public:
    SiPixelLorentzAngleByRegionComparisonTwoTags() : SiPixelLorentzAngleByRegionComparisonBase() { setTwoTags(true); }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelLorentzAngle) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValue);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValueComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValueComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleByRegionComparisonTwoTags);
}
