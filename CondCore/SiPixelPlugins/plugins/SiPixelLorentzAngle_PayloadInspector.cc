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
#include "CondCore/SiPixelPlugins/interface/PixelRegionContainers.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"

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

  using namespace cond::payloadInspector;

  /************************************************
    1d histogram of SiPixelLorentzAngle of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiPixelLorentzAngleValue : public Histogram1D<SiPixelLorentzAngle, SINGLE_IOV> {
  public:
    SiPixelLorentzAngleValue()
        : Histogram1D<SiPixelLorentzAngle, SINGLE_IOV>(
              "SiPixel LorentzAngle values", "SiPixel LorentzAngle values", 100, 0.0, 0.1) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const &iov : tag.iovs) {
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
  class SiPixelLorentzAngleValues : public PlotImage<SiPixelLorentzAngle, SINGLE_IOV> {
  public:
    SiPixelLorentzAngleValues() : PlotImage<SiPixelLorentzAngle, SINGLE_IOV>("SiPixelLorentzAngle Values") {}

    bool fill() override {
      gStyle->SetOptStat("emr");

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelLorentzAngle> payload = fetchPayload(std::get<1>(iov));
      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();
      auto extrema = SiPixelPI::findMinMaxInMap(LAMap_);

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>("value",
                                       "SiPixel LA value;SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T];# modules",
                                       50,
                                       extrema.first * 0.9,
                                       extrema.second * 1.1);

      SiPixelPI::adjustCanvasMargins(canvas.cd(), 0.06, 0.12, 0.12, 0.05);
      canvas.Modified();

      for (const auto &element : LAMap_) {
        h1->Fill(element.second);
      }

      h1->SetTitle("");
      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetFillColor(kRed);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("bar2");

      SiPixelPI::makeNicePlotStyle(h1.get());
      h1->SetStats(true);

      canvas.Update();

      TLegend legend = TLegend(0.40, 0.88, 0.94, 0.93);
      legend.SetHeader(("Payload hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      //legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.SetLineColor(10);
      legend.Draw("same");

      TPaveStats *st = (TPaveStats *)h1->FindObject("stats");
      st->SetTextSize(0.03);
      SiPixelPI::adjustStats(st, 0.15, 0.83, 0.39, 0.93);

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
    1d histogram of SiPixelLorentzAngle of 1 IOV per region
  *************************************************/
  template <bool isBarrel>
  class SiPixelLorentzAngleValuesPerRegion : public PlotImage<SiPixelLorentzAngle, SINGLE_IOV> {
  public:
    SiPixelLorentzAngleValuesPerRegion()
        : PlotImage<SiPixelLorentzAngle, SINGLE_IOV>("SiPixelLorentzAngle Values per region") {}

    bool fill() override {
      gStyle->SetOptStat("emr");

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelLorentzAngle> payload = fetchPayload(std::get<1>(iov));
      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();
      auto extrema = SiPixelPI::findMinMaxInMap(LAMap_);

      TCanvas canvas("Canv", "Canv", isBarrel ? 1400 : 1800, 1200);
      canvas.cd();

      SiPixelPI::PhaseInfo phaseInfo(LAMap_.size());
      const char *path_toTopologyXML = phaseInfo.pathToTopoXML();

      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto myPlots = PixelRegions::PixelRegionContainers(&tTopo, phaseInfo.phase());
      myPlots.bookAll("SiPixel LA",
                      "SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T]",
                      "#modules",
                      50,
                      extrema.first * 0.9,
                      extrema.second * 1.1);

      canvas.Modified();

      for (const auto &element : LAMap_) {
        myPlots.fill(element.first, element.second);
      }

      myPlots.beautify();
      myPlots.draw(canvas, isBarrel);

      TLegend legend = TLegend(0.40, 0.88, 0.93, 0.90);
      legend.SetHeader(("Hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      //legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.SetLineColor(10);

      unsigned int maxPads = canvas.GetListOfPrimitives()->GetSize();  //= isBarrel ? 4 : 12;
      for (unsigned int c = 1; c <= maxPads; c++) {
        if (phaseInfo.phase() == SiPixelPI::phase::two && (c == 5 || c == 10))
          continue;
        canvas.cd(c);
        SiPixelPI::adjustCanvasMargins(canvas.cd(c), 0.06, 0.12, 0.12, 0.05);
        legend.Draw("same");
        canvas.cd(c)->Update();
      }

      myPlots.stats();

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);

      int index = 0;
      for (unsigned int c = 1; c <= maxPads; c++) {
        if (phaseInfo.phase() == SiPixelPI::phase::two && (c == 5 || c == 10))
          continue;
        canvas.cd(c);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (PixelRegions::getIDLabels(phaseInfo.phase(), isBarrel)[index] +
                          ", IOV:" + std::to_string(std::get<0>(iov)))
                             .c_str());

        index++;
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  using SiPixelLorentzAngleValuesBarrel = SiPixelLorentzAngleValuesPerRegion<true>;
  using SiPixelLorentzAngleValuesEndcap = SiPixelLorentzAngleValuesPerRegion<false>;

  /************************************************
    1d histogram of SiPixelLorentzAngle of 2 IOV per region
  *************************************************/
  template <bool isBarrel, IOVMultiplicity nIOVs, int ntags>
  class SiPixelLorentzAngleValuesComparisonPerRegion : public PlotImage<SiPixelLorentzAngle, nIOVs, ntags> {
  public:
    SiPixelLorentzAngleValuesComparisonPerRegion()
        : PlotImage<SiPixelLorentzAngle, nIOVs, ntags>(
              Form("SiPixelLorentzAngle Values Comparisons per region %i tag(s)", ntags)) {}

    bool fill() override {
      gStyle->SetOptStat("emr");

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto f_tagname = PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        l_tagname = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiPixelLorentzAngle> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::map<uint32_t, float> l_LAMap_ = last_payload->getLorentzAngles();
      auto l_extrema = SiPixelPI::findMinMaxInMap(l_LAMap_);

      std::shared_ptr<SiPixelLorentzAngle> first_payload = this->fetchPayload(std::get<1>(firstiov));
      std::map<uint32_t, float> f_LAMap_ = first_payload->getLorentzAngles();
      auto f_extrema = SiPixelPI::findMinMaxInMap(f_LAMap_);

      auto max = (l_extrema.second > f_extrema.second) ? l_extrema.second : f_extrema.second;
      auto min = (l_extrema.first < f_extrema.first) ? l_extrema.first : f_extrema.first;

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Canv", "Canv", isBarrel ? 1400 : 1800, 1200);
      canvas.cd();

      SiPixelPI::PhaseInfo l_phaseInfo(l_LAMap_.size());
      SiPixelPI::PhaseInfo f_phaseInfo(f_LAMap_.size());

      if (l_phaseInfo.isComparedWithPhase2(f_phaseInfo)) {
        SiPixelPI::displayNotSupported(canvas, std::max(f_LAMap_.size(), l_LAMap_.size()));
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      // deal with last IOV
      const char *path_toTopologyXML = l_phaseInfo.pathToTopoXML();

      auto l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto l_myPlots = PixelRegions::PixelRegionContainers(&l_tTopo, l_phaseInfo.phase());
      l_myPlots.bookAll("SiPixel LA,last",
                        "SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T]",
                        "#modules",
                        50,
                        min * 0.9,
                        max * 1.1);

      for (const auto &element : l_LAMap_) {
        l_myPlots.fill(element.first, element.second);
      }

      l_myPlots.beautify();
      l_myPlots.draw(canvas, isBarrel, "bar2", f_phaseInfo.isPhase1Comparison(l_phaseInfo));

      // deal with first IOV
      path_toTopologyXML = f_phaseInfo.pathToTopoXML();

      auto f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto f_myPlots = PixelRegions::PixelRegionContainers(&f_tTopo, f_phaseInfo.phase());
      f_myPlots.bookAll("SiPixel LA,first",
                        "SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T]",
                        "#modules",
                        50,
                        min * 0.9,
                        max * 1.1);

      for (const auto &element : f_LAMap_) {
        f_myPlots.fill(element.first, element.second);
      }

      f_myPlots.beautify(kAzure, kBlue);
      f_myPlots.draw(canvas, isBarrel, "HISTsames", f_phaseInfo.isPhase1Comparison(l_phaseInfo));

      // rescale the y-axis ranges in order to fit the canvas
      l_myPlots.rescaleMax(f_myPlots);

      // done dealing with IOVs

      auto colorTag = PixelRegions::L1;  //: PixelRegions::Rm1l;
      std::unique_ptr<TLegend> legend;
      if (this->m_plotAnnotations.ntags == 2) {
        legend = std::make_unique<TLegend>(0.36, 0.86, 0.94, 0.92);
        legend->AddEntry(l_myPlots.getHistoFromMap(colorTag).get(), ("#color[2]{" + l_tagname + "}").c_str(), "F");
        legend->AddEntry(f_myPlots.getHistoFromMap(colorTag).get(), ("#color[4]{" + f_tagname + "}").c_str(), "F");
        legend->SetTextSize(0.024);
      } else {
        legend = std::make_unique<TLegend>(0.58, 0.80, 0.90, 0.92);
        legend->AddEntry(l_myPlots.getHistoFromMap(colorTag).get(), ("#color[2]{" + lastIOVsince + "}").c_str(), "F");
        legend->AddEntry(f_myPlots.getHistoFromMap(colorTag).get(), ("#color[4]{" + firstIOVsince + "}").c_str(), "F");
        legend->SetTextSize(0.040);
      }
      legend->SetLineColor(10);

      unsigned int maxPads = canvas.GetListOfPrimitives()->GetSize();  //= isBarrel ? 4 : 12;
      for (unsigned int c = 1; c <= maxPads; c++) {
        if (l_phaseInfo.phase() == SiPixelPI::phase::two && (c == 5 || c == 10))
          continue;
        canvas.cd(c);
        SiPixelPI::adjustCanvasMargins(canvas.cd(c), 0.06, 0.12, 0.12, 0.05);
        legend->Draw("same");
        canvas.cd(c)->Update();
      }

      f_myPlots.stats(0);
      l_myPlots.stats(1);

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);

      int index = 0;
      for (unsigned int c = 1; c <= maxPads; c++) {
        if (l_phaseInfo.phase() == SiPixelPI::phase::two && (c == 5 || c == 10))
          continue;
        canvas.cd(c);

        COUT << "c:" << c << " index:" << index << " : "
             << PixelRegions::getIDLabels(l_phaseInfo.phase(), isBarrel)[index] << "\n";

        ltx.DrawLatexNDC(
            gPad->GetLeftMargin(),
            1 - gPad->GetTopMargin() + 0.01,
            (PixelRegions::getIDLabels(l_phaseInfo.phase(), isBarrel)[index] + " : #color[4]{" +
             std::to_string(std::get<0>(firstiov)) + "} vs #color[2]{" + std::to_string(std::get<0>(lastiov)) + "}")
                .c_str());
        index++;
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

#ifdef MMDEBUG
      canvas.SaveAs("DEBUG.root");
#endif

      return true;
    }
  };

  using SiPixelLorentzAngleValuesBarrelCompareSingleTag =
      SiPixelLorentzAngleValuesComparisonPerRegion<true, MULTI_IOV, 1>;
  using SiPixelLorentzAngleValuesEndcapCompareSingleTag =
      SiPixelLorentzAngleValuesComparisonPerRegion<false, MULTI_IOV, 1>;

  using SiPixelLorentzAngleValuesBarrelCompareTwoTags =
      SiPixelLorentzAngleValuesComparisonPerRegion<true, SINGLE_IOV, 2>;
  using SiPixelLorentzAngleValuesEndcapCompareTwoTags =
      SiPixelLorentzAngleValuesComparisonPerRegion<false, SINGLE_IOV, 2>;

  /************************************************
    1d histogram of SiPixelLorentzAngle comparisons
  *************************************************/
  template <IOVMultiplicity nIOVs, int ntags>
  class SiPixelLorentzAngleValueComparisonBase : public PlotImage<SiPixelLorentzAngle, nIOVs, ntags> {
  public:
    SiPixelLorentzAngleValueComparisonBase()
        : PlotImage<SiPixelLorentzAngle, nIOVs, ntags>(Form("SiPixelLorentzAngle Values Comparison %i tag(s)", ntags)) {
    }

    bool fill() override {
      TH1F::SetDefaultSumw2(true);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto f_tagname = PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        l_tagname = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiPixelLorentzAngle> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::map<uint32_t, float> l_LAMap_ = last_payload->getLorentzAngles();
      auto l_extrema = SiPixelPI::findMinMaxInMap(l_LAMap_);

      std::shared_ptr<SiPixelLorentzAngle> first_payload = this->fetchPayload(std::get<1>(firstiov));
      std::map<uint32_t, float> f_LAMap_ = first_payload->getLorentzAngles();
      auto f_extrema = SiPixelPI::findMinMaxInMap(f_LAMap_);

      auto max = (l_extrema.second > f_extrema.second) ? l_extrema.second : f_extrema.second;
      auto min = (l_extrema.first < f_extrema.first) ? l_extrema.first : f_extrema.first;

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto hfirst =
          std::make_unique<TH1F>("value_first",
                                 "SiPixel LA value;SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T];# modules",
                                 50,
                                 min * 0.9,
                                 max * 1.1);
      hfirst->SetStats(false);

      auto hlast =
          std::make_unique<TH1F>("value_last",
                                 "SiPixel LA value;SiPixel LorentzAngle #mu_{H}(tan#theta_{L}/B) [1/T];# modules",
                                 50,
                                 min * 0.9,
                                 max * 1.1);
      hlast->SetStats(false);

      SiPixelPI::adjustCanvasMargins(canvas.cd(), 0.06, 0.12, 0.12, 0.05);
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
      hfirst->SetBarWidth(0.95);
      hfirst->Draw("histbar");

      hlast->SetTitle("");
      hlast->SetFillColorAlpha(kBlue, 0.20);
      hlast->SetBarWidth(0.95);
      hlast->Draw("histbarsame");

      SiPixelPI::makeNicePlotStyle(hfirst.get());
      SiPixelPI::makeNicePlotStyle(hlast.get());

      canvas.Update();

      TLegend legend = TLegend(0.30, 0.86, 0.95, 0.94);
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
      ltx.SetTextSize(0.047);
      ltx.SetTextAlign(11);
      std::string ltxText;
      if (this->m_plotAnnotations.ntags == 2) {
        ltxText = fmt::sprintf("#color[2]{%s, %s} vs #color[4]{%s, %s}",
                               f_tagname,
                               std::to_string(std::get<0>(firstiov)),
                               l_tagname,
                               std::to_string(std::get<0>(lastiov)));
      } else {
        ltxText = fmt::sprintf("%s IOV: #color[2]{%s} vs IOV: #color[4]{%s}",
                               f_tagname,
                               std::to_string(std::get<0>(firstiov)),
                               std::to_string(std::get<0>(lastiov)));
      }
      ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.01, ltxText.c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  using SiPixelLorentzAngleValueComparisonSingleTag = SiPixelLorentzAngleValueComparisonBase<MULTI_IOV, 1>;
  using SiPixelLorentzAngleValueComparisonTwoTags = SiPixelLorentzAngleValueComparisonBase<SINGLE_IOV, 2>;

  /************************************************
   Summary Comparison per region of SiPixelLorentzAngle between 2 IOVs
  *************************************************/
  template <IOVMultiplicity nIOVs, int ntags>
  class SiPixelLorentzAngleByRegionComparisonBase : public PlotImage<SiPixelLorentzAngle, nIOVs, ntags> {
  public:
    SiPixelLorentzAngleByRegionComparisonBase()
        : PlotImage<SiPixelLorentzAngle, nIOVs, ntags>(
              Form("SiPixelLorentzAngle Comparison by Region %i tag(s)", ntags)) {}

    bool fill() override {
      gStyle->SetPaintTextFormat(".3f");

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto f_tagname = PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        l_tagname = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiPixelLorentzAngle> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::map<uint32_t, float> l_LAMap_ = last_payload->getLorentzAngles();
      std::shared_ptr<SiPixelLorentzAngle> first_payload = this->fetchPayload(std::get<1>(firstiov));
      std::map<uint32_t, float> f_LAMap_ = first_payload->getLorentzAngles();

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      SiPixelPI::PhaseInfo l_phaseInfo(l_LAMap_.size());
      SiPixelPI::PhaseInfo f_phaseInfo(f_LAMap_.size());

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

      // deal with first IOV
      const char *path_toTopologyXML = f_phaseInfo.pathToTopoXML();

      auto f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

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
        t_info_fromXML.fillGeometryInfo(detid, f_tTopo, f_phaseInfo.phase());

        SiPixelPI::regions thePart = t_info_fromXML.filterThePartition();
        if (thePart != SiPixelPI::NUM_OF_REGIONS) {
          FirstLA_spectraByRegion[thePart]->Fill(it.second);
        }
      }  // ends loop on the vector of error transforms

      // deal with last IOV
      path_toTopologyXML = l_phaseInfo.pathToTopoXML();

      auto l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

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
        t_info_fromXML.fillGeometryInfo(detid, l_tTopo, l_phaseInfo.phase());

        SiPixelPI::regions thePart = t_info_fromXML.filterThePartition();
        if (thePart != SiPixelPI::NUM_OF_REGIONS) {
          LastLA_spectraByRegion[thePart]->Fill(it.second);
        }
      }  // ends loop on the vector of error transforms

      // fill the summary plots
      int bin = 1;
      for (int r = SiPixelPI::BPixL1o; r != SiPixelPI::NUM_OF_REGIONS; r++) {
        SiPixelPI::regions part = static_cast<SiPixelPI::regions>(r);

        summaryFirst->GetXaxis()->SetBinLabel(bin, SiPixelPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float f_mean =
            FirstLA_spectraByRegion[part]->GetMean() > 10.e-6 ? FirstLA_spectraByRegion[part]->GetMean() : 0.;
        summaryFirst->SetBinContent(bin, f_mean);
        //summaryFirst->SetBinError(bin,LA_spectraByRegion[hash]->GetRMS());

        summaryLast->GetXaxis()->SetBinLabel(bin, SiPixelPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float l_mean = LastLA_spectraByRegion[part]->GetMean() > 10.e-6 ? LastLA_spectraByRegion[part]->GetMean() : 0.;
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

      summaryFirst->Draw("b text0");
      summaryLast->Draw("b text0 same");

      TLegend legend = TLegend(0.52, 0.80, 0.98, 0.9);
      legend.SetHeader("#mu_{H} value comparison", "C");  // option "C" allows to center the header
      std::string l_tagOrHash, f_tagOrHash;
      if (this->m_plotAnnotations.ntags == 2) {
        l_tagOrHash = l_tagname;
        f_tagOrHash = f_tagname;
      } else {
        l_tagOrHash = std::get<1>(lastiov);
        f_tagOrHash = std::get<1>(firstiov);
      }

      legend.AddEntry(
          summaryLast.get(),
          ("IOV: #scale[1.2]{" + std::to_string(std::get<0>(lastiov)) + "} | #color[4]{" + l_tagOrHash + "}").c_str(),
          "F");
      legend.AddEntry(
          summaryFirst.get(),
          ("IOV: #scale[1.2]{" + std::to_string(std::get<0>(firstiov)) + "} | #color[2]{" + f_tagOrHash + "}").c_str(),
          "F");

      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }
  };

  using SiPixelLorentzAngleByRegionComparisonSingleTag = SiPixelLorentzAngleByRegionComparisonBase<MULTI_IOV, 1>;
  using SiPixelLorentzAngleByRegionComparisonTwoTags = SiPixelLorentzAngleByRegionComparisonBase<SINGLE_IOV, 2>;

  /************************************************
   occupancy style map Pixel LA
  *************************************************/

  template <SiPixelPI::DetType myType>
  class SiPixelLorentzAngleMap : public PlotImage<SiPixelLorentzAngle, SINGLE_IOV> {
  public:
    SiPixelLorentzAngleMap()
        : PlotImage<SiPixelLorentzAngle, SINGLE_IOV>("SiPixelLorentzAngle Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiPixelLorentzAngle> payload = fetchPayload(std::get<1>(iov));

      Phase1PixelROCMaps thePixLAMap("");

      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();
      if (LAMap_.size() != SiPixelPI::phase1size) {
        edm::LogError("SiPixelLorentzAngle_PayloadInspector")
            << "SiPixelLorentzAngle maps are not supported for non-Phase1 Pixel geometries !";
        TCanvas canvas("Canv", "Canv", 1200, 1000);
        SiPixelPI::displayNotSupported(canvas, LAMap_.size());
        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      // hard-coded phase-I
      std::array<double, n_layers> b_minima = {{999., 999., 999., 999.}};
      std::array<double, n_rings> f_minima = {{999., 999.}};

      for (const auto &element : LAMap_) {
        int subid = DetId(element.first).subdetId();
        if (subid == PixelSubdetector::PixelBarrel) {
          auto layer = m_trackerTopo.pxbLayer(DetId(element.first));
          if (element.second < b_minima.at(layer - 1)) {
            b_minima.at(layer - 1) = element.second;
          }
        } else if (subid == PixelSubdetector::PixelEndcap) {
          auto ring = SiPixelPI::ring(DetId(element.first), m_trackerTopo, true);
          if (element.second < f_minima.at(ring - 1)) {
            f_minima.at(ring - 1) = element.second;
          }
        }
        thePixLAMap.fillWholeModule(element.first, element.second);
      }

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, k_height[myType]);
      canvas.cd();

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      std::string IOVstring = (unpacked.first == 0)
                                  ? std::to_string(unpacked.second)
                                  : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second));

      const auto headerText = fmt::sprintf("#color[4]{%s},  IOV: #color[4]{%s}", tagname, IOVstring);

      switch (myType) {
        case SiPixelPI::t_barrel:
          thePixLAMap.drawBarrelMaps(canvas, headerText);
          break;
        case SiPixelPI::t_forward:
          thePixLAMap.drawForwardMaps(canvas, headerText);
          break;
        case SiPixelPI::t_all:
          thePixLAMap.drawMaps(canvas, headerText);
          break;
        default:
          throw cms::Exception("SiPixelLorentzAngleMap") << "\nERROR: unrecognized Pixel Detector part " << std::endl;
      }

      if (myType == SiPixelPI::t_barrel || myType == SiPixelPI::t_all) {
        // set the minima and maxima of the barrel
        for (unsigned int lay = 1; lay <= n_layers; lay++) {
          auto h_bpix_LA = thePixLAMap.getLayerMaps();

          COUT << " layer:" << lay << " max:" << h_bpix_LA[lay - 1]->GetMaximum() << " min: " << b_minima.at(lay - 1)
               << std::endl;

          h_bpix_LA[lay - 1]->GetZaxis()->SetRangeUser(b_minima.at(lay - 1) - 0.001,
                                                       h_bpix_LA[lay - 1]->GetMaximum() + 0.001);
        }
      }

      if (myType == SiPixelPI::t_forward || myType == SiPixelPI::t_all) {
        // set the minima and maxima of the endcaps
        for (unsigned int ring = 1; ring <= n_rings; ring++) {
          auto h_fpix_LA = thePixLAMap.getRingMaps();

          COUT << " ringer:" << ring << " max:" << h_fpix_LA[ring - 1]->GetMaximum()
               << " min: " << f_minima.at(ring - 1) << std::endl;

          h_fpix_LA[ring - 1]->GetZaxis()->SetRangeUser(f_minima.at(ring - 1) - 0.001,
                                                        h_fpix_LA[ring - 1]->GetMaximum() + 0.001);
        }
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outPixLA.root");
#endif

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
    static constexpr std::array<int, 3> k_height = {{1200, 600, 1600}};
    static constexpr int n_layers = 4;
    static constexpr int n_rings = 2;
  };

  using SiPixelBPixLorentzAngleMap = SiPixelLorentzAngleMap<SiPixelPI::t_barrel>;
  using SiPixelFPixLorentzAngleMap = SiPixelLorentzAngleMap<SiPixelPI::t_forward>;
  using SiPixelFullLorentzAngleMapByROC = SiPixelLorentzAngleMap<SiPixelPI::t_all>;

  /************************************************
   Full Pixel Tracker Map class
  *************************************************/
  class SiPixelLorentzAngleFullPixelMap : public PlotImage<SiPixelLorentzAngle, SINGLE_IOV> {
  public:
    SiPixelLorentzAngleFullPixelMap() : PlotImage<SiPixelLorentzAngle, SINGLE_IOV>("SiPixelLorentzAngle Map") {
      label_ = "SiPixelLorentzAngleFullPixelMap";
      payloadString = "Lorentz Angle";
    }

    bool fill() override {
      gStyle->SetPalette(1);
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelLorentzAngle> payload = this->fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        Phase1PixelSummaryMap fullMap(
            "", fmt::sprintf("%s", payloadString), fmt::sprintf("%s #mu_{H} [1/T]", payloadString));
        fullMap.createTrackerBaseMap();

        std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();

        if (LAMap_.size() == SiPixelPI::phase0size || LAMap_.size() > SiPixelPI::phase1size) {
          edm::LogError(label_) << "There are " << LAMap_.size()
                                << " DetIds in this payload. SiPixelLorentzAngleFullPixelMap maps are not supported "
                                   "for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, LAMap_.size());
          std::string fileName(this->m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        } else {
          if (LAMap_.size() < SiPixelPI::phase1size) {
            edm::LogWarning(label_) << "\n ********* WARNING! ********* \n There are " << LAMap_.size()
                                    << " DetIds in this payload !"
                                    << "\n **************************** \n";
          }
        }

        for (const auto &entry : LAMap_) {
          fullMap.fillTrackerMap(entry.first, entry.second);
        }

        TCanvas canvas("Canv", "Canv", 3000, 2000);
        fullMap.printTrackerMap(canvas);

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.025);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin() + 0.01,
            gPad->GetBottomMargin() + 0.01,
            ("#color[4]{" + tag.name + "}, IOV: #color[4]{" + std::to_string(std::get<0>(iov)) + "}").c_str());

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

  protected:
    std::string payloadString;
    std::string label_;
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelLorentzAngle) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValue);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValuesBarrelCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValuesEndcapCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValuesBarrelCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValuesEndcapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValueComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValueComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleByRegionComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixLorentzAngleMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixLorentzAngleMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFullLorentzAngleMapByROC);
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleFullPixelMap);
}
