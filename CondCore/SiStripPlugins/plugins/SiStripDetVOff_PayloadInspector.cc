// system includes
#include <iostream>
#include <memory>
#include <sstream>
#include <fmt/printf.h>

// user includes
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

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

  class SiStripDetVOff_LV : public TimeHistoryPlot<SiStripDetVOff, int> {
  public:
    SiStripDetVOff_LV() : TimeHistoryPlot<SiStripDetVOff, int>("Nr of mod with LV OFF vs time", "nLVOff") {}

    int getFromPayload(SiStripDetVOff& payload) override { return payload.getLVoffCounts(); }
  };

  class SiStripDetVOff_HV : public TimeHistoryPlot<SiStripDetVOff, int> {
  public:
    SiStripDetVOff_HV() : TimeHistoryPlot<SiStripDetVOff, int>("Nr of mod with HV OFF vs time", "nHVOff") {}

    int getFromPayload(SiStripDetVOff& payload) override { return payload.getHVoffCounts(); }
  };

  namespace SiStripDetVOffPI {
    enum type { t_LV = 0, t_HV = 1, t_V };
  }

  /************************************************
    Templated TrackerMap of Module L- H- L||V Voff
  *************************************************/
  template <SiStripDetVOffPI::type my_type>
  class SiStripDetVOff_TrackerMapBase : public PlotImage<SiStripDetVOff, SINGLE_IOV> {
  public:
    SiStripDetVOff_TrackerMapBase() : PlotImage<SiStripDetVOff, SINGLE_IOV>("Tracker Map: Is Module VOff") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      unsigned long IOVsince = std::get<0>(iov);
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload(std::get<1>(iov));

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripIsModuleVOff");
      tmap->setPalette(1);
      std::string titleMap{};

      switch (my_type) {
        case SiStripDetVOffPI::t_LV: {
          titleMap = fmt::sprintf("TrackerMap of LV VOff modules | Tag: %s | IOV: %s", tagname, getIOVsince(IOVsince));
          break;
        }
        case SiStripDetVOffPI::t_HV: {
          titleMap = fmt::sprintf("TrackerMap of HV VOff modules | Tag: %s | IOV: %s", tagname, getIOVsince(IOVsince));
          break;
        }
        case SiStripDetVOffPI::t_V: {
          titleMap =
              fmt::sprintf("TrackerMap of VOff modules (HV or LV) | Tag: %s | IOV: %s", tagname, getIOVsince(IOVsince));
          break;
        }
        default:
          edm::LogError("SiStripDetVOff_IsModuleVOff_TrackerMap") << "Unrecognized type: " << my_type << std::endl;
          break;
      }

      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto& d : detid) {
        if ((payload->IsModuleLVOff(d) && (my_type == SiStripDetVOffPI::t_LV)) ||
            (payload->IsModuleHVOff(d) && (my_type == SiStripDetVOffPI::t_HV)) ||
            (payload->IsModuleVOff(d) && (my_type == SiStripDetVOffPI::t_V))) {
          tmap->fill(d, 1.);
        }
      }  // loop over detIds

      std::string fileName(m_imageFileName);
      //tmap->save_as_HVtrackermap(true, 0., 1.01, fileName); // not working ?
      tmap->save(true, 0., 1.01, fileName);

      return true;
    }

  private:
    const char* getIOVsince(const unsigned long IOV) {
      int run = 0;
      static char buf[256];

      if (IOV < 4294967296) {  // run type IOV
        run = IOV;
        std::sprintf(buf, "%d", run);
      } else {  // time type IOV
        run = IOV >> 32;
        time_t t = run;
        struct tm lt;
        localtime_r(&t, &lt);
        strftime(buf, sizeof(buf), "%F %R:%S", &lt);
        buf[sizeof(buf) - 1] = 0;
      }
      return buf;
    }
  };

  using SiStripDetVOff_IsModuleVOff_TrackerMap = SiStripDetVOff_TrackerMapBase<SiStripDetVOffPI::t_V>;
  using SiStripDetVOff_IsModuleLVOff_TrackerMap = SiStripDetVOff_TrackerMapBase<SiStripDetVOffPI::t_LV>;
  using SiStripDetVOff_IsModuleHVOff_TrackerMap = SiStripDetVOff_TrackerMapBase<SiStripDetVOffPI::t_HV>;

  /************************************************
    List of unpowered modules
  *************************************************/
  template <SiStripDetVOffPI::type my_type>
  class SiStripDetVOffListOfModules : public Histogram1DD<SiStripDetVOff, SINGLE_IOV> {
  public:
    SiStripDetVOffListOfModules()
        : Histogram1DD<SiStripDetVOff, SINGLE_IOV>(
              "SiStrip Off modules", "SiStrip Off modules", 15148, 0., 15148., "DetId of VOff module") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripDetVOff> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);
          int i = 0;  // count modules

          //std::cout.precision(1);

          for (const auto& d : detid) {
            switch (my_type) {
              case SiStripDetVOffPI::t_LV: {
                if (payload->IsModuleLVOff(d)) {
                  //std::cout << "is LV: " << i << " " << std::fixed << double(d) << std::endl;
                  fillWithBinAndValue(i, double(d));
                }
                break;
              }
              case SiStripDetVOffPI::t_HV: {
                if (payload->IsModuleHVOff(d)) {
                  //std::cout << "is HV: " << i << " " << std::fixed << double(d) << std::endl;
                  fillWithBinAndValue(i, double(d));
                }
                break;
              }
              case SiStripDetVOffPI::t_V: {
                if (payload->IsModuleVOff(d)) {
                  //std::cout << "is V: " << i << " " << std::fixed << double(d) << std::endl;
                  fillWithBinAndValue(i, double(d));
                }
                break;
              }
              default:
                edm::LogError("SiStripDetVOffListOfModules") << "Unrecognized type: " << my_type << std::endl;
                break;
            }     // switch
            i++;  // increase counting of modules
          }       // loop on detids
        }         // if gets the payload
      }           // loop on iovs
      return true;
    }  // fill()
  };

  using SiStripVOffListOfModules = SiStripDetVOffListOfModules<SiStripDetVOffPI::t_V>;
  using SiStripLVOffListOfModules = SiStripDetVOffListOfModules<SiStripDetVOffPI::t_LV>;
  using SiStripHVOffListOfModules = SiStripDetVOffListOfModules<SiStripDetVOffPI::t_HV>;

  /************************************************
    test class
  *************************************************/

  class SiStripDetVOffTest : public Histogram1D<SiStripDetVOff, SINGLE_IOV> {
  public:
    SiStripDetVOffTest()
        : Histogram1D<SiStripDetVOff, SINGLE_IOV>("SiStrip DetVOff test", "SiStrip DetVOff test", 10, 0.0, 10.0),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripDetVOff> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          SiStripDetSummary summaryHV{&m_trackerTopo};
          SiStripDetSummary summaryLV{&m_trackerTopo};

          for (const auto& d : detid) {
            if (payload->IsModuleLVOff(d))
              summaryLV.add(d);
            if (payload->IsModuleHVOff(d))
              summaryHV.add(d);
          }
          std::map<unsigned int, SiStripDetSummary::Values> mapHV = summaryHV.getCounts();
          std::map<unsigned int, SiStripDetSummary::Values> mapLV = summaryLV.getCounts();

          // SiStripPI::printSummary(mapHV);
          // SiStripPI::printSummary(mapLV);

          std::stringstream ss;
          ss << "Summary of HV off detectors:" << std::endl;
          summaryHV.print(ss, true);

          ss << "Summary of LV off detectors:" << std::endl;
          summaryLV.print(ss, true);

          std::cout << ss.str() << std::endl;

        }  // payload
      }    // iovs
      return true;
    }  // fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Plot DetVOff by region 
  *************************************************/

  class SiStripDetVOffByRegion : public PlotImage<SiStripDetVOff, SINGLE_IOV> {
  public:
    SiStripDetVOffByRegion()
        : PlotImage<SiStripDetVOff, SINGLE_IOV>("SiStrip DetVOff By Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload(std::get<1>(iov));

      unsigned long IOV = std::get<0>(iov);
      int run = 0;
      if (IOV < 4294967296) {
        run = std::get<0>(iov);
      } else {  // time type IOV
        run = IOV >> 32;
      }

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryHV{&m_trackerTopo};
      SiStripDetSummary summaryLV{&m_trackerTopo};

      for (const auto& d : detid) {
        if (payload->IsModuleLVOff(d))
          summaryLV.add(d);
        if (payload->IsModuleHVOff(d))
          summaryHV.add(d);
      }
      std::map<unsigned int, SiStripDetSummary::Values> mapHV = summaryHV.getCounts();
      std::map<unsigned int, SiStripDetSummary::Values> mapLV = summaryLV.getCounts();
      std::vector<unsigned int> keys;
      std::transform(
          mapHV.begin(),
          mapHV.end(),
          std::back_inserter(keys),
          [](const std::map<unsigned int, SiStripDetSummary::Values>::value_type& pair) { return pair.first; });

      //=========================

      TCanvas canvas("DetVOff Partion summary", "SiStripDetVOff region summary", 1200, 1000);
      canvas.cd();
      auto h_HV = std::make_unique<TH1F>(
          "HVbyRegion", "SiStrip HV/LV summary by region;; modules with HV off", mapHV.size(), 0., mapHV.size());
      auto h_LV = std::make_unique<TH1F>(
          "LVbyRegion", "SiStrip HV/LV summary by region;; modules with LV off", mapLV.size(), 0., mapLV.size());

      h_HV->SetStats(false);
      h_LV->SetStats(false);

      h_HV->SetTitle(nullptr);
      h_LV->SetTitle(nullptr);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.10);
      canvas.SetRightMargin(0.10);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto& index : keys) {
        iBin++;
        int countHV = mapHV[index].count;
        int countLV = mapLV[index].count;

        if (currentDetector.empty())
          currentDetector = "TIB";

        switch ((index) / 1000) {
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

        h_HV->SetBinContent(iBin, countHV);
        h_HV->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(index).second);
        h_HV->GetXaxis()->LabelsOption("v");

        h_LV->SetBinContent(iBin, countLV);
        h_LV->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(index).second);
        h_LV->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      auto extrema = SiStripPI::getExtrema(h_LV.get(), h_HV.get());
      h_HV->GetYaxis()->SetRangeUser(extrema.first, extrema.second);
      h_LV->GetYaxis()->SetRangeUser(extrema.first, extrema.second);

      h_HV->SetMarkerStyle(20);
      h_HV->SetMarkerSize(1);
      h_HV->SetLineColor(kRed);
      h_HV->SetMarkerColor(kRed);
      h_HV->Draw("HIST");
      h_HV->Draw("TEXT45same");

      h_LV->SetMarkerStyle(21);
      h_LV->SetMarkerSize(1);
      h_LV->SetLineColor(kBlue);
      h_LV->SetLineStyle(9);
      h_LV->SetMarkerColor(kBlue);
      h_LV->Draw("HISTsame");
      h_LV->Draw("TEXT45same");

      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(
            h_HV->GetBinLowEdge(line), canvas.cd()->GetUymin(), h_HV->GetBinLowEdge(line), canvas.cd()->GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.45, 0.80, 0.90, 0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
      legend.AddEntry(h_HV.get(), ("HV channels: " + std::to_string(payload->getHVoffCounts())).c_str(), "PL");
      legend.AddEntry(h_LV.get(), ("LV channels: " + std::to_string(payload->getLVoffCounts())).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      if (IOV < 4294967296)
        t1.DrawLatex(0.5, 0.96, Form("SiStrip DetVOff, IOV %i", run));
      else {  // time type IOV
        time_t t = run;
        char buf[256];
        struct tm lt;
        localtime_r(&t, &lt);
        strftime(buf, sizeof(buf), "%F %R:%S", &lt);
        buf[sizeof(buf) - 1] = 0;
        t1.DrawLatex(0.5, 0.96, Form("SiStrip DetVOff, IOV %s", buf));
      }

      // Remove the current axis
      h_HV.get()->GetYaxis()->SetLabelOffset(999);
      h_HV.get()->GetYaxis()->SetTickLength(0);
      h_HV.get()->GetYaxis()->SetTitleOffset(999);

      h_LV.get()->GetYaxis()->SetLabelOffset(999);
      h_LV.get()->GetYaxis()->SetTickLength(0);
      h_LV.get()->GetYaxis()->SetTitleOffset(999);

      //draw an axis on the left side
      auto l_axis = std::make_unique<TGaxis>(
          gPad->GetUxmin(), gPad->GetUymin(), gPad->GetUxmin(), gPad->GetUymax(), 0, extrema.second, 510);
      l_axis->SetLineColor(kRed);
      l_axis->SetTextColor(kRed);
      l_axis->SetLabelColor(kRed);
      l_axis->SetTitleOffset(1.2);
      l_axis->SetTitleColor(kRed);
      l_axis->SetTitle(h_HV.get()->GetYaxis()->GetTitle());
      l_axis->Draw();

      //draw an axis on the right side
      auto r_axis = std::make_unique<TGaxis>(
          gPad->GetUxmax(), gPad->GetUymin(), gPad->GetUxmax(), gPad->GetUymax(), 0, extrema.second, 510, "+L");
      r_axis->SetLineColor(kBlue);
      r_axis->SetTextColor(kBlue);
      r_axis->SetLabelColor(kBlue);
      r_axis->SetTitleColor(kBlue);
      r_axis->SetTitleOffset(1.2);
      r_axis->SetTitle(h_LV.get()->GetYaxis()->GetTitle());
      r_axis->Draw();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripDetVOff) {
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOff_LV);
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOff_HV);
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOff_IsModuleVOff_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOff_IsModuleLVOff_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOff_IsModuleHVOff_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOffTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripVOffListOfModules);
  PAYLOAD_INSPECTOR_CLASS(SiStripLVOffListOfModules);
  PAYLOAD_INSPECTOR_CLASS(SiStripHVOffListOfModules);
  PAYLOAD_INSPECTOR_CLASS(SiStripDetVOffByRegion);
}
