/*!
  \file SiStripLorentzAngle_PayloadInspector
  \Payload Inspector Plugin for SiStrip Lorentz angles
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/09/21 10:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include <memory>
#include <sstream>

// include ROOT
#include "TH2F.h"
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
    1d histogram of SiStripLorentzAngle of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripLorentzAngleValue : public cond::payloadInspector::Histogram1D<SiStripLorentzAngle> {
  public:
    SiStripLorentzAngleValue()
        : cond::payloadInspector::Histogram1D<SiStripLorentzAngle>(
              "SiStrip LorentzAngle values", "SiStrip LorentzAngle values", 100, 0.0, 0.05) {
      Base::setSingleIov(true);
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const &iov : tag.iovs) {
        std::shared_ptr<SiStripLorentzAngle> payload = Base::fetchPayload(std::get<1>(iov));
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
    TrackerMap of SiStrip Lorentz Angle
  *************************************************/
  class SiStripLorentzAngle_TrackerMap : public cond::payloadInspector::PlotImage<SiStripLorentzAngle> {
  public:
    SiStripLorentzAngle_TrackerMap()
        : cond::payloadInspector::PlotImage<SiStripLorentzAngle>("Tracker Map SiStrip Lorentz Angle") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripLorentzAngle> payload = fetchPayload(std::get<1>(iov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripLorentzAngle");
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of SiStrip Lorentz Angle per module, payload : " + std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();

      for (const auto &element : LAMap_) {
        tmap->fill(element.first, element.second);
      }  // loop over the LA MAP

      std::pair<float, float> extrema = tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);

      // protect against uniform values (LA values are defined positive)
      if (extrema.first != extrema.second) {
        tmap->save(true, 0, 0, fileName);
      } else {
        tmap->save(true, extrema.first * 0.95, extrema.first * 1.05, fileName);
      }

      return true;
    }
  };

  /************************************************
    Plot Lorentz Angle averages by partition 
  *************************************************/

  class SiStripLorentzAngleByRegion : public cond::payloadInspector::PlotImage<SiStripLorentzAngle> {
  public:
    SiStripLorentzAngleByRegion()
        : cond::payloadInspector::PlotImage<SiStripLorentzAngle>("SiStripLorentzAngle By Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripLorentzAngle> payload = fetchPayload(std::get<1>(iov));

      SiStripDetSummary summaryLA{&m_trackerTopo};

      std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();

      for (const auto &element : LAMap_) {
        summaryLA.add(element.first, element.second);
      }

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryLA.getCounts();
      //=========================

      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>("byRegion",
                                       "SiStrip LA average by partition;; average SiStrip Lorentz Angle [rad]",
                                       map.size(),
                                       0.,
                                       map.size());
      h1->SetStats(false);
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.17);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : map) {
        iBin++;
        int count = element.second.count;
        double mean = (element.second.mean) / count;

        if (currentDetector.empty())
          currentDetector = "TIB";

        switch ((element.first) / 1000) {
          case 1:
            detector = "TIB";
            break;
          case 2:
            detector = "TOB";
            break;
          case 3:
            detector = "TEC";
            break;
          case 4:
            detector = "TID";
            break;
        }

        h1->SetBinContent(iBin, mean);
        h1->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        h1->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("HIST");
      h1->Draw("Psame");

      canvas.Update();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto &line : boundaries) {
        l[i] = TLine(h1->GetBinLowEdge(line), canvas.GetUymin(), h1->GetBinLowEdge(line), canvas.GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
      legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Plot SiStripLorentz Angle averages by partition comparison
  *************************************************/

  template <int ntags, cond::payloadInspector::IOVMultiplicity nIOVs>
  class SiStripLorentzAngleComparatorByRegionBase
      : public cond::payloadInspector::PlotImage<SiStripLorentzAngle, nIOVs, ntags> {
  public:
    SiStripLorentzAngleComparatorByRegionBase()
        : cond::payloadInspector::PlotImage<SiStripLorentzAngle, nIOVs, ntags>(
              "SiStripLorentzAngle By Region Comparison"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto tagname1 = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        tagname2 = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripLorentzAngle> f_payload = this->fetchPayload(std::get<1>(firstiov));
      std::shared_ptr<SiStripLorentzAngle> l_payload = this->fetchPayload(std::get<1>(lastiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      //=========================
      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.cd();
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.03);
      canvas.SetTopMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      std::shared_ptr<TH1F> h_first;
      std::shared_ptr<TH1F> h_last;

      fillTheHistogram(f_payload, h_first, boundaries, 0);
      fillTheHistogram(l_payload, h_last, boundaries, 1);

      canvas.cd();
      h_first->Draw("HIST");
      h_last->Draw("Psame");

      canvas.Update();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto &line : boundaries) {
        l[i] = TLine(h_first->GetBinLowEdge(line), canvas.GetUymin(), h_first->GetBinLowEdge(line), canvas.GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.045);
      ltx.SetTextAlign(11);

      std::unique_ptr<TLegend> legend = std::make_unique<TLegend>(0.50, 0.25, 0.80, 0.35);
      if (this->m_plotAnnotations.ntags == 2) {
        legend->AddEntry(h_last.get(), ("#color[2]{" + tagname2 + "}").c_str(), "P");
        legend->AddEntry(h_first.get(), ("#color[4]{" + tagname1 + "}").c_str(), "L");
        legend->SetTextSize(0.024);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         ("IOV : #color[4]{" + std::to_string(std::get<0>(firstiov)) + "} vs #color[2]{" +
                          std::to_string(std::get<0>(lastiov)) + "}")
                             .c_str());
      } else {
        legend->AddEntry(h_last.get(), ("IOV: #color[2]{" + lastIOVsince + "}").c_str(), "P");
        legend->AddEntry(h_first.get(), ("IOV: #color[4]{" + firstIOVsince + "}").c_str(), "L");
        legend->SetTextSize(0.040);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.01, ("Tag: " + tagname1).c_str());

        legend->SetLineColor(kBlack);
      }
      legend->Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;

    void fillTheHistogram(const std::shared_ptr<SiStripLorentzAngle> &payload,
                          std::shared_ptr<TH1F> &hist,
                          std::vector<int> &boundaries,
                          unsigned int index = 0) {
      SiStripDetSummary summaryLA{&m_trackerTopo};
      auto LAMap_ = payload->getLorentzAngles();
      for (const auto &element : LAMap_) {
        summaryLA.add(element.first, element.second);
      }

      auto map = summaryLA.getCounts();
      hist = std::make_shared<TH1F>(
          (Form("byRegion_%i", index)), ";; average SiStrip Lorentz Angle #mu_{H} [1/T]", map.size(), 0., map.size());

      hist->SetStats(false);
      if (index == 0) {
        hist->SetLineColor(kBlue);
        hist->SetMarkerColor(kBlue);
        hist->SetLineWidth(2);
        hist->SetMarkerStyle(kFourSquaresX);
      } else {
        hist->SetMarkerStyle(kFourSquaresX);
        hist->SetLineColor(kRed);
        hist->SetMarkerColor(kRed);
      }
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : map) {
        iBin++;
        int count = element.second.count;
        double mean = (element.second.mean) / count;

        if (currentDetector.empty())
          currentDetector = "TIB";

        switch ((element.first) / 1000) {
          case 1:
            detector = "TIB";
            break;
          case 2:
            detector = "TOB";
            break;
          case 3:
            detector = "TEC";
            break;
          case 4:
            detector = "TID";
            break;
        }

        hist->SetBinContent(iBin, mean);
        hist->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        hist->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          if (index == 0) {
            boundaries.push_back(iBin);
          }
          currentDetector = detector;
        }
      }

      hist->GetYaxis()->SetTitleSize(0.04);
      hist->GetYaxis()->SetTitleOffset(1.55);
      hist->GetYaxis()->CenterTitle(true);
      hist->GetYaxis()->SetRangeUser(0., hist->GetMaximum() * 1.30);
      hist->SetMarkerSize(2);
    }
  };

  using SiStripLorentzAngleByRegionCompareSingleTag =
      SiStripLorentzAngleComparatorByRegionBase<1, cond::payloadInspector::MULTI_IOV>;
  using SiStripLorentzAngleByRegionCompareTwoTags =
      SiStripLorentzAngleComparatorByRegionBase<2, cond::payloadInspector::SINGLE_IOV>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripLorentzAngle) {
  PAYLOAD_INSPECTOR_CLASS(SiStripLorentzAngleValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripLorentzAngle_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripLorentzAngleByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripLorentzAngleByRegionCompareSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripLorentzAngleByRegionCompareTwoTags);
}
