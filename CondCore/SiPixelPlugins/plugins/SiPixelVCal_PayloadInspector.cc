/*!
  \file SiPixelVCal_PayloadInspector
  \Payload Inspector Plugin for SiPixel Lorentz angles
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2019/06/20 10:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondCore/SiPixelPlugins/interface/PixelRegionContainers.h"

#include <memory>
#include <sstream>
#include <fmt/printf.h>

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

  namespace SiPixelVCalPI {
    enum type { t_slope = 0, t_offset = 1 };
  }

  /************************************************
    1d histogram of SiPixelVCal of 1 IOV 
  *************************************************/
  // inherit from one of the predefined plot class: Histogram1D
  template <SiPixelVCalPI::type myType>
  class SiPixelVCalValue : public Histogram1D<SiPixelVCal, SINGLE_IOV> {
  public:
    SiPixelVCalValue()
        : Histogram1D<SiPixelVCal, SINGLE_IOV>("SiPixel VCal values",
                                               "SiPixel VCal values",
                                               100,
                                               myType == SiPixelVCalPI::t_slope ? 40. : -700,
                                               myType == SiPixelVCalPI::t_slope ? 60. : 0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const &iov : tag.iovs) {
        std::shared_ptr<SiPixelVCal> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          auto VCalMap_ = payload->getSlopeAndOffset();
          for (const auto &element : VCalMap_) {
            if (myType == SiPixelVCalPI::t_slope) {
              fillWithValue(element.second.slope);
            } else if (myType == SiPixelVCalPI::t_offset) {
              fillWithValue(element.second.offset);
            }
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  using SiPixelVCalSlopeValue = SiPixelVCalValue<SiPixelVCalPI::t_slope>;
  using SiPixelVCalOffsetValue = SiPixelVCalValue<SiPixelVCalPI::t_offset>;

  /************************************************
    1d histogram of SiPixelVCal of 1 IOV 
  *************************************************/
  class SiPixelVCalValues : public PlotImage<SiPixelVCal, SINGLE_IOV> {
  public:
    SiPixelVCalValues() : PlotImage<SiPixelVCal, SINGLE_IOV>("SiPixelVCal Values") {}

    bool fill() override {
      gStyle->SetOptStat("emr");

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelVCal> payload = fetchPayload(std::get<1>(iov));
      auto VCalMap_ = payload->getSlopeAndOffset();

      auto slopes = payload->getAllSlopes();
      auto offsets = payload->getAllOffsets();

      /*
      std::transform(VCalMap_.begin(),
		     VCalMap_.end(),
		     std::inserter(slopes,slopes.end()),
		     [](std::pair<unsigned int, SiPixelVCal::VCal> vcalentry) -> std::pair<uint32_t,float> { return std::make_pair(vcalentry.first,vcalentry.second.slope); });
      
      */

      auto s_extrema = SiPixelPI::findMinMaxInMap(slopes);
      auto o_extrema = SiPixelPI::findMinMaxInMap(offsets);

      auto o_range = (o_extrema.second - o_extrema.first) / 10.;

      TCanvas canvas("Canv", "Canv", 1000, 1000);
      canvas.Divide(1, 2);
      canvas.cd();
      auto h_slope =
          std::make_shared<TH1F>("slope value",
                                 "SiPixel VCal slope value;SiPixel VCal slope value [ADC/VCal units];# modules",
                                 50,
                                 s_extrema.first * 0.9,
                                 s_extrema.second * 1.1);

      auto h_offset = std::make_shared<TH1F>("offset value",
                                             "SiPixel VCal offset value;SiPixel VCal offset value [ADC];# modules",
                                             50,
                                             o_extrema.first - o_range,
                                             o_extrema.second + o_range);

      for (unsigned int i = 1; i <= 2; i++) {
        SiPixelPI::adjustCanvasMargins(canvas.cd(i), 0.06, 0.12, 0.08, 0.03);
      }

      for (const auto &slope : slopes) {
        h_slope->Fill(slope.second);
      }

      for (const auto &offset : offsets) {
        h_offset->Fill(offset.second);
      }

      canvas.cd(1);
      adjustHisto(h_slope);
      canvas.Update();

      TLegend legend = TLegend(0.40, 0.83, 0.94, 0.93);
      legend.SetHeader(("Payload hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      legend.AddEntry(h_slope.get(), ("TAG: " + tag.name).c_str(), "F");
      legend.SetTextSize(0.035);
      legend.SetLineColor(10);
      legend.Draw("same");

      TPaveStats *st = (TPaveStats *)h_slope->FindObject("stats");
      st->SetTextSize(0.035);
      SiPixelPI::adjustStats(st, 0.15, 0.83, 0.30, 0.93);

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel VCal Slope IOV:" + std::to_string(std::get<0>(iov))).c_str());

      canvas.cd(2);
      adjustHisto(h_offset);
      canvas.Update();
      legend.Draw("same");

      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel VCal Offset IOV:" + std::to_string(std::get<0>(iov))).c_str());

      TPaveStats *st2 = (TPaveStats *)h_offset->FindObject("stats");
      st2->SetTextSize(0.035);
      SiPixelPI::adjustStats(st2, 0.15, 0.83, 0.30, 0.93);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    void adjustHisto(const std::shared_ptr<TH1F> &histo) {
      histo->SetTitle("");
      histo->GetYaxis()->SetRangeUser(0., histo->GetMaximum() * 1.30);
      histo->SetFillColor(kRed);
      histo->SetMarkerStyle(20);
      histo->SetMarkerSize(1);
      histo->Draw("bar2");
      SiPixelPI::makeNicePlotStyle(histo.get());
      histo->SetStats(true);
      histo->GetYaxis()->SetTitleOffset(0.9);
    }
  };

  /************************************************
      1d histogram of SiPixelVCal of 1 IOV per region
  *************************************************/
  template <bool isBarrel, SiPixelVCalPI::type myType>
  class SiPixelVCalValuesPerRegion : public PlotImage<SiPixelVCal, SINGLE_IOV> {
  public:
    SiPixelVCalValuesPerRegion() : PlotImage<SiPixelVCal, SINGLE_IOV>("SiPixelVCal Values per region") {}

    bool fill() override {
      gStyle->SetOptStat("emr");

      auto tag = PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelVCal> payload = fetchPayload(std::get<1>(iov));
      SiPixelVCal::mapToDetId Map_;
      if (myType == SiPixelVCalPI::t_slope) {
        Map_ = payload->getAllSlopes();
      } else {
        Map_ = payload->getAllOffsets();
      }
      auto extrema = SiPixelPI::findMinMaxInMap(Map_);
      auto range = (extrema.second - extrema.first) / 10.;

      TCanvas canvas("Canv", "Canv", isBarrel ? 1400 : 1800, 1200);
      if (Map_.size() > SiPixelPI::phase1size) {
        SiPixelPI::displayNotSupported(canvas, Map_.size());
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      canvas.cd();

      SiPixelPI::PhaseInfo phaseInfo(Map_.size());
      const char *path_toTopologyXML = phaseInfo.pathToTopoXML();

      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto myPlots = PixelRegions::PixelRegionContainers(&tTopo, phaseInfo.phase());
      myPlots.bookAll((myType == SiPixelVCalPI::t_slope) ? "SiPixel VCal slope value" : "SiPixel VCal offset value",
                      (myType == SiPixelVCalPI::t_slope) ? "SiPixel VCal slope value [ADC/VCal units]"
                                                         : "SiPixel VCal offset value [ADC]",
                      "#modules",
                      50,
                      extrema.first - range,
                      extrema.second + range);

      canvas.Modified();

      for (const auto &element : Map_) {
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

      unsigned int maxPads = isBarrel ? 4 : 12;
      for (unsigned int c = 1; c <= maxPads; c++) {
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

      for (unsigned int c = 1; c <= maxPads; c++) {
        auto index = isBarrel ? c - 1 : c + 3;

        canvas.cd(c);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         fmt::sprintf("%s, #color[2]{%s, IOV: %s}",
                                      PixelRegions::IDlabels.at(index),
                                      tagname,
                                      std::to_string(std::get<0>(iov)))
                             .c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  using SiPixelVCalSlopeValuesBarrel = SiPixelVCalValuesPerRegion<true, SiPixelVCalPI::t_slope>;
  using SiPixelVCalSlopeValuesEndcap = SiPixelVCalValuesPerRegion<false, SiPixelVCalPI::t_slope>;

  using SiPixelVCalOffsetValuesBarrel = SiPixelVCalValuesPerRegion<true, SiPixelVCalPI::t_offset>;
  using SiPixelVCalOffsetValuesEndcap = SiPixelVCalValuesPerRegion<false, SiPixelVCalPI::t_offset>;

  /************************************************
      1d histogram of SiPixelVCal (slope or offset) of 2 IOV per region
  *************************************************/
  template <bool isBarrel, SiPixelVCalPI::type myType, IOVMultiplicity nIOVs, int ntags>
  class SiPixelVCalValuesCompareSubdet : public PlotImage<SiPixelVCal, nIOVs, ntags> {
  public:
    SiPixelVCalValuesCompareSubdet()
        : PlotImage<SiPixelVCal, nIOVs, ntags>(Form("SiPixelVCal Values Comparisons by Subdet %i tags(s)", ntags)) {}

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

      SiPixelVCal::mapToDetId l_Map_;
      SiPixelVCal::mapToDetId f_Map_;

      std::shared_ptr<SiPixelVCal> last_payload = this->fetchPayload(std::get<1>(lastiov));
      if (myType == SiPixelVCalPI::t_slope) {
        l_Map_ = last_payload->getAllSlopes();
      } else {
        l_Map_ = last_payload->getAllOffsets();
      }
      auto l_extrema = SiPixelPI::findMinMaxInMap(l_Map_);

      std::shared_ptr<SiPixelVCal> first_payload = this->fetchPayload(std::get<1>(firstiov));
      if (myType == SiPixelVCalPI::t_slope) {
        f_Map_ = first_payload->getAllSlopes();
      } else {
        f_Map_ = first_payload->getAllOffsets();
      }

      auto f_extrema = SiPixelPI::findMinMaxInMap(f_Map_);

      auto max = (l_extrema.second > f_extrema.second) ? l_extrema.second : f_extrema.second;
      auto min = (l_extrema.first < f_extrema.first) ? l_extrema.first : f_extrema.first;

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Canv", "Canv", isBarrel ? 1400 : 1800, 1200);
      if ((f_Map_.size() > SiPixelPI::phase1size) || (l_Map_.size() > SiPixelPI::phase1size)) {
        SiPixelPI::displayNotSupported(canvas, std::max(f_Map_.size(), l_Map_.size()));
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      canvas.cd();

      SiPixelPI::PhaseInfo l_phaseInfo(l_Map_.size());
      SiPixelPI::PhaseInfo f_phaseInfo(f_Map_.size());

      // deal with last IOV

      const char *path_toTopologyXML = l_phaseInfo.pathToTopoXML();

      auto l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto delta = std::abs(max - min) / 10.;

      auto l_myPlots = PixelRegions::PixelRegionContainers(&l_tTopo, l_phaseInfo.phase());
      l_myPlots.bookAll(
          fmt::sprintf("SiPixel VCal %s,last", (myType == SiPixelVCalPI::t_slope ? "slope" : "offset")),
          fmt::sprintf("SiPixel VCal %s", (myType == SiPixelVCalPI::t_slope ? " slope [ADC/VCal]" : " offset [ADC]")),
          "#modules",
          50,
          min - delta,
          max + delta);

      for (const auto &element : l_Map_) {
        l_myPlots.fill(element.first, element.second);
      }

      l_myPlots.beautify();
      l_myPlots.draw(canvas, isBarrel, "bar2", true);

      // deal with first IOV

      path_toTopologyXML = f_phaseInfo.pathToTopoXML();

      auto f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto f_myPlots = PixelRegions::PixelRegionContainers(&f_tTopo, f_phaseInfo.phase());
      f_myPlots.bookAll(
          fmt::sprintf("SiPixel VCal %s,first", (myType == SiPixelVCalPI::t_slope ? "slope" : "offset")),
          fmt::sprintf("SiPixel VCal %s", (myType == SiPixelVCalPI::t_slope ? " slope [ADC/VCal]" : " offset [ADC]")),
          "#modules",
          50,
          min - delta,
          max + delta);

      for (const auto &element : f_Map_) {
        f_myPlots.fill(element.first, element.second);
      }

      f_myPlots.beautify(kAzure, kBlue);
      f_myPlots.draw(canvas, isBarrel, "HISTsames", true);

      // rescale the y-axis ranges in order to fit the canvas
      l_myPlots.rescaleMax(f_myPlots);

      // done dealing with IOVs

      auto colorTag = isBarrel ? PixelRegions::L1 : PixelRegions::Rm1l;
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

      unsigned int maxPads = isBarrel ? 4 : 12;
      for (unsigned int c = 1; c <= maxPads; c++) {
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

      for (unsigned int c = 1; c <= maxPads; c++) {
        auto index = isBarrel ? c - 1 : c + 3;
        canvas.cd(c);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (PixelRegions::IDlabels.at(index) + " : #color[4]{" + std::to_string(std::get<0>(firstiov)) +
                          "} vs #color[2]{" + std::to_string(std::get<0>(lastiov)) + "}")
                             .c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  using SiPixelVCalSlopesBarrelCompareSingleTag =
      SiPixelVCalValuesCompareSubdet<true, SiPixelVCalPI::t_slope, MULTI_IOV, 1>;
  using SiPixelVCalOffsetsBarrelCompareSingleTag =
      SiPixelVCalValuesCompareSubdet<true, SiPixelVCalPI::t_offset, MULTI_IOV, 1>;

  using SiPixelVCalSlopesEndcapCompareSingleTag =
      SiPixelVCalValuesCompareSubdet<false, SiPixelVCalPI::t_slope, MULTI_IOV, 1>;
  using SiPixelVCalOffsetsEndcapCompareSingleTag =
      SiPixelVCalValuesCompareSubdet<false, SiPixelVCalPI::t_offset, MULTI_IOV, 1>;

  using SiPixelVCalSlopesBarrelCompareTwoTags =
      SiPixelVCalValuesCompareSubdet<true, SiPixelVCalPI::t_slope, SINGLE_IOV, 2>;
  using SiPixelVCalOffsetsBarrelCompareTwoTags =
      SiPixelVCalValuesCompareSubdet<true, SiPixelVCalPI::t_offset, SINGLE_IOV, 2>;

  using SiPixelVCalSlopesEndcapCompareTwoTags =
      SiPixelVCalValuesCompareSubdet<false, SiPixelVCalPI::t_slope, SINGLE_IOV, 2>;
  using SiPixelVCalOffsetsEndcapCompareTwoTags =
      SiPixelVCalValuesCompareSubdet<false, SiPixelVCalPI::t_offset, SINGLE_IOV, 2>;

  /************************************************
      1d histogram of SiPixelVCal of 1 IOV
  *************************************************/

  template <SiPixelVCalPI::type myType, IOVMultiplicity nIOVs, int ntags>
  class SiPixelVCalValueComparisonBase : public PlotImage<SiPixelVCal, nIOVs, ntags> {
  public:
    SiPixelVCalValueComparisonBase()
        : PlotImage<SiPixelVCal, nIOVs, ntags>(Form("SiPixelVCal Synoptic Values Comparison %i tag(s)", ntags)) {}

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

      SiPixelVCal::mapToDetId l_Map_;
      SiPixelVCal::mapToDetId f_Map_;

      std::shared_ptr<SiPixelVCal> last_payload = this->fetchPayload(std::get<1>(lastiov));
      if (myType == SiPixelVCalPI::t_slope) {
        l_Map_ = last_payload->getAllSlopes();
      } else {
        l_Map_ = last_payload->getAllOffsets();
      }
      auto l_extrema = SiPixelPI::findMinMaxInMap(l_Map_);

      std::shared_ptr<SiPixelVCal> first_payload = this->fetchPayload(std::get<1>(firstiov));
      if (myType == SiPixelVCalPI::t_slope) {
        f_Map_ = first_payload->getAllSlopes();
      } else {
        f_Map_ = first_payload->getAllOffsets();
      }
      auto f_extrema = SiPixelPI::findMinMaxInMap(f_Map_);

      auto max = (l_extrema.second > f_extrema.second) ? l_extrema.second : f_extrema.second;
      auto min = (l_extrema.first < f_extrema.first) ? l_extrema.first : f_extrema.first;
      auto delta = std::abs(max - min) / 10.;

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto hfirst = std::unique_ptr<TH1F>(
          new TH1F("value_first",
                   fmt::sprintf("SiPixel VCal %s value;SiPixel VCal %s ;# modules",
                                (myType == SiPixelVCalPI::t_slope ? "slope" : "offset"),
                                (myType == SiPixelVCalPI::t_slope ? " slope [ADC/VCal]" : " offset [ADC]"))
                       .c_str(),
                   50,
                   min - delta,
                   max + delta));
      hfirst->SetStats(false);

      auto hlast = std::unique_ptr<TH1F>(
          new TH1F("value_last",
                   fmt::sprintf("SiPixel VCal %s value;SiPixel VCal %s ;# modules",
                                (myType == SiPixelVCalPI::t_slope ? "slope" : "offset"),
                                (myType == SiPixelVCalPI::t_slope ? " slope [ADC/VCal]" : " offset [ADC]"))
                       .c_str(),
                   50,
                   min - delta,
                   max + delta));
      hlast->SetStats(false);

      SiPixelPI::adjustCanvasMargins(canvas.cd(), 0.06, 0.12, 0.12, 0.05);
      canvas.Modified();

      for (const auto &element : f_Map_) {
        hfirst->Fill(element.second);
      }

      for (const auto &element : l_Map_) {
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
      legend.AddEntry(hfirst.get(), ("payload: #color[2]{" + std::get<1>(firstiov) + "}").c_str(), "F");
      legend.AddEntry(hlast.get(), ("payload: #color[4]{" + std::get<1>(lastiov) + "}").c_str(), "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.040);
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

  using SiPixelVCalSlopesComparisonSingleTag = SiPixelVCalValueComparisonBase<SiPixelVCalPI::t_slope, MULTI_IOV, 1>;
  using SiPixelVCalOffsetsComparisonSingleTag = SiPixelVCalValueComparisonBase<SiPixelVCalPI::t_offset, MULTI_IOV, 1>;
  using SiPixelVCalSlopesComparisonTwoTags = SiPixelVCalValueComparisonBase<SiPixelVCalPI::t_slope, SINGLE_IOV, 2>;
  using SiPixelVCalOffsetsComparisonTwoTags = SiPixelVCalValueComparisonBase<SiPixelVCalPI::t_offset, SINGLE_IOV, 2>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelVCal) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopeValue);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetValue);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopeValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopeValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopesBarrelCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetsBarrelCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopesEndcapCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetsEndcapCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopesBarrelCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetsBarrelCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopesEndcapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetsEndcapCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopesComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetsComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopesComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetsComparisonTwoTags);
}
