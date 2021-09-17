/*!
  \file SiStripBackPlaneCorrection_PayloadInspector
  \Payload Inspector Plugin for SiStrip Backplane corrections
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/09/21 10:01:03 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"

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

  using namespace cond::payloadInspector;

  /************************************************
    1d histogram of SiStripBackPlaneCorrection of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripBackPlaneCorrectionValue : public Histogram1D<SiStripBackPlaneCorrection, SINGLE_IOV> {
  public:
    SiStripBackPlaneCorrectionValue()
        : Histogram1D<SiStripBackPlaneCorrection, SINGLE_IOV>(
              "SiStrip BackPlaneCorrection values", "SiStrip BackPlaneCorrection values", 100, 0.0, 0.1) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const &iov : tag.iovs) {
        std::shared_ptr<SiStripBackPlaneCorrection> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::map<uint32_t, float> BPMap_ = payload->getBackPlaneCorrections();

          for (const auto &element : BPMap_) {
            fillWithValue(element.second);
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
    TrackerMap of SiStrip BackPlane Correction
  *************************************************/
  class SiStripBackPlaneCorrection_TrackerMap : public PlotImage<SiStripBackPlaneCorrection, SINGLE_IOV> {
  public:
    SiStripBackPlaneCorrection_TrackerMap()
        : PlotImage<SiStripBackPlaneCorrection, SINGLE_IOV>("Tracker Map SiStrip Backplane correction") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripBackPlaneCorrection> payload = fetchPayload(std::get<1>(iov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripBackPlaneCorrection");
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of SiStrip BP correction per module, payload : " + std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::map<uint32_t, float> BPMap_ = payload->getBackPlaneCorrections();

      for (const auto &element : BPMap_) {
        tmap->fill(element.first, element.second);
      }  // loop over the BP MAP

      std::pair<float, float> extrema = tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);

      // protect against uniform values across the map (BP corrections are defined positive)
      if (extrema.first != extrema.second) {
        tmap->save(true, 0, 0, fileName);
      } else {
        tmap->save(true, extrema.first * 0.95, extrema.first * 1.05, fileName);
      }

      return true;
    }
  };

  /************************************************
    Plot SiStrip BackPlane Correction averages by partition 
  *************************************************/

  class SiStripBackPlaneCorrectionByRegion : public PlotImage<SiStripBackPlaneCorrection, SINGLE_IOV> {
  public:
    SiStripBackPlaneCorrectionByRegion()
        : PlotImage<SiStripBackPlaneCorrection, SINGLE_IOV>("SiStripBackPlaneCorrection By Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripBackPlaneCorrection> payload = fetchPayload(std::get<1>(iov));

      SiStripDetSummary summaryBP{&m_trackerTopo};

      std::map<uint32_t, float> BPMap_ = payload->getBackPlaneCorrections();

      for (const auto &element : BPMap_) {
        summaryBP.add(element.first, element.second);
      }

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryBP.getCounts();
      //=========================

      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.cd();
      auto h1 = std::make_unique<TH1F>(
          "byRegion",
          "SiStrip Backplane correction average by partition;; average SiStrip BackPlane Correction",
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

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripBackPlaneCorrection) {
  PAYLOAD_INSPECTOR_CLASS(SiStripBackPlaneCorrectionValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripBackPlaneCorrection_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBackPlaneCorrectionByRegion);
}
