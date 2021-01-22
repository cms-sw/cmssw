/*!
  \file SiStripBadStrip_PayloadInspector
  \Payload Inspector Plugin for SiStrip Bad Strip
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/08/14 14:37:22 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/SiStripPlugins/interface/SiStripTkMaps.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    test class
  *************************************************/

  class SiStripBadStripTest : public Histogram1D<SiStripBadStrip, SINGLE_IOV> {
  public:
    SiStripBadStripTest()
        : Histogram1D<SiStripBadStrip, SINGLE_IOV>("SiStrip Bad Strip test", "SiStrip Bad Strip test", 10, 0.0, 10.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripBadStrip> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          std::stringstream ss;
          ss << "Summary of bad strips:" << std::endl;

          //payload->printDebug(ss);
          //payload->printSummary(ss);
          //std::cout<<ss.str()<<std::endl;

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            SiStripBadStrip::Range range = payload->getRange(d);
            for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second;
                 ++badStrip) {
              ss << "DetId=" << d << " Strip=" << payload->decode(*badStrip).firstStrip << ":"
                 << payload->decode(*badStrip).range << " flag=" << payload->decode(*badStrip).flag << std::endl;
            }
          }

          std::cout << ss.str() << std::endl;

        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
    TrackerMap of SiStripBadStrip (bad strip per detid)
  *************************************************/
  class SiStripBadModuleTrackerMap : public PlotImage<SiStripBadStrip, SINGLE_IOV> {
  public:
    SiStripBadModuleTrackerMap() : PlotImage<SiStripBadStrip, SINGLE_IOV>("Tracker Map of SiStrip Bad Strips") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = PlotBase::getTag<0>().name;
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload(std::get<1>(iov));

      auto theIOVsince = std::to_string(std::get<0>(iov));

      std::string titleMap = "Modules w/ at least 1 bad Strip, Run: " + theIOVsince + " (tag: " + tagname + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripBadStrips");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto& d : detid) {
        tmap->fill(d, 1.);
      }  // loop over detIds

      //=========================

      std::string fileName(m_imageFileName);
      tmap->save(true, 0, 1., fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripBadStrip (bad strips fraction)
  *************************************************/
  class SiStripBadStripFractionTrackerMap : public PlotImage<SiStripBadStrip, SINGLE_IOV> {
  public:
    SiStripBadStripFractionTrackerMap()
        : PlotImage<SiStripBadStrip, SINGLE_IOV>("Tracker Map of SiStrip Bad Components fraction") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = PlotBase::getTag<0>().name;
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload(std::get<1>(iov));

      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      auto theIOVsince = std::to_string(std::get<0>(iov));

      std::string titleMap = "Fraction of bad Strips per module, Run: " + theIOVsince + " (tag: " + tagname + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripBadStrips");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload->getRange(d);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStripsPerDetId[d] += payload->decode(*badStrip).range;
          //ss << "DetId="<< d << " Strip=" << payload->decode(*badStrip).firstStrip <<":"<< payload->decode(*badStrip).range << " flag="<< payload->decode(*badStrip).flag << std::endl;
        }
        float fraction = badStripsPerDetId[d] / (128. * reader->getNumberOfApvsAndStripLength(d).first);
        tmap->fill(d, fraction);
      }  // loop over detIds

      //=========================

      std::pair<float, float> extrema = tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);

      // protect against uniform values across the map (bad components fractions are defined positive)
      if (extrema.first != extrema.second) {
        tmap->save(true, 0, 0, fileName);
      } else {
        tmap->save(true, extrema.first * 0.95, extrema.first * 1.05, fileName);
      }

      delete reader;
      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripBadStrip (bad strips fraction)
  *************************************************/
  class SiStripBadStripFractionTkMap : public PlotImage<SiStripBadStrip, SINGLE_IOV> {
  public:
    SiStripBadStripFractionTkMap()
        : PlotImage<SiStripBadStrip, SINGLE_IOV>("Tracker Map of SiStrip Bad Components fraction") {}

    bool fill() override {
      //SiStripPI::setPaletteStyle(SiStripPI::DEFAULT);
      gStyle->SetPalette(1);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = PlotBase::getTag<0>().name;
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload(std::get<1>(iov));

      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      auto theIOVsince = std::to_string(std::get<0>(iov));

      std::string titleMap =
          "Fraction of bad Strips per module, Run: " + theIOVsince + " (tag:#color[2]{" + tagname + "})";

      SiStripTkMaps myMap("COLZA0 L");
      myMap.bookMap(titleMap, "Fraction of bad Strips per module");

      SiStripTkMaps ghost("AL");
      ghost.bookMap(titleMap, "");

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload->getRange(d);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStripsPerDetId[d] += payload->decode(*badStrip).range;
        }
        float fraction = badStripsPerDetId[d] / (128. * reader->getNumberOfApvsAndStripLength(d).first);
        if (fraction > 0.) {
          myMap.fill(d, fraction);
        }
      }  // loop over detIds

      //=========================

      std::string fileName(m_imageFileName);
      TCanvas canvas("Bad Components fraction", "bad components fraction");
      myMap.drawMap(canvas, "");
      ghost.drawMap(canvas, "same");
      canvas.SaveAs(fileName.c_str());

      delete reader;
      return true;
    }
  };

  /************************************************
    time history histogram of bad components fraction
  *************************************************/

  class SiStripBadStripFractionByRun : public HistoryPlot<SiStripBadStrip, float> {
  public:
    SiStripBadStripFractionByRun()
        : HistoryPlot<SiStripBadStrip, float>("SiStrip Bad Strip fraction per run", "Bad Strip fraction [%]") {}
    ~SiStripBadStripFractionByRun() override = default;

    float getFromPayload(SiStripBadStrip& payload) override {
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload.getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += payload.decode(*badStrip).range;
        }
        badStripsPerDetId[d] = badStrips;
      }  // loop over detIds

      float numerator(0.), denominator(0.);
      std::vector<uint32_t> all_detids = reader->getAllDetIds();
      for (const auto& det : all_detids) {
        denominator += 128. * reader->getNumberOfApvsAndStripLength(det).first;
        if (badStripsPerDetId.count(det) != 0)
          numerator += badStripsPerDetId[det];
      }

      delete reader;
      return (numerator / denominator) * 100.;

    }  // payload
  };

  /************************************************
    time history histogram of bad components fraction (TIB)
  *************************************************/

  class SiStripBadStripTIBFractionByRun : public HistoryPlot<SiStripBadStrip, float> {
  public:
    SiStripBadStripTIBFractionByRun()
        : HistoryPlot<SiStripBadStrip, float>("SiStrip Inner Barrel Bad Strip fraction per run",
                                              "TIB Bad Strip fraction [%]") {}
    ~SiStripBadStripTIBFractionByRun() override = default;

    float getFromPayload(SiStripBadStrip& payload) override {
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload.getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += payload.decode(*badStrip).range;
        }
        badStripsPerDetId[d] = badStrips;
      }  // loop over detIds

      float numerator(0.), denominator(0.);
      std::vector<uint32_t> all_detids = reader->getAllDetIds();
      for (const auto& det : all_detids) {
        int subid = DetId(det).subdetId();
        if (subid != StripSubdetector::TIB)
          continue;
        denominator += 128. * reader->getNumberOfApvsAndStripLength(det).first;
        if (badStripsPerDetId.count(det) != 0)
          numerator += badStripsPerDetId[det];
      }

      delete reader;
      return (numerator / denominator) * 100.;

    }  // payload
  };

  /************************************************
    time history histogram of bad components fraction (TOB)
  *************************************************/

  class SiStripBadStripTOBFractionByRun : public HistoryPlot<SiStripBadStrip, float> {
  public:
    SiStripBadStripTOBFractionByRun()
        : HistoryPlot<SiStripBadStrip, float>("SiStrip Outer Barrel Bad Strip fraction per run",
                                              "TOB Bad Strip fraction [%]") {}
    ~SiStripBadStripTOBFractionByRun() override = default;

    float getFromPayload(SiStripBadStrip& payload) override {
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload.getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += payload.decode(*badStrip).range;
        }
        badStripsPerDetId[d] = badStrips;
      }  // loop over detIds

      float numerator(0.), denominator(0.);
      std::vector<uint32_t> all_detids = reader->getAllDetIds();
      for (const auto& det : all_detids) {
        int subid = DetId(det).subdetId();
        if (subid != StripSubdetector::TOB)
          continue;
        denominator += 128. * reader->getNumberOfApvsAndStripLength(det).first;
        if (badStripsPerDetId.count(det) != 0)
          numerator += badStripsPerDetId[det];
      }

      delete reader;
      return (numerator / denominator) * 100.;

    }  // payload
  };

  /************************************************
    time history histogram of bad components fraction (TID)
   *************************************************/

  class SiStripBadStripTIDFractionByRun : public HistoryPlot<SiStripBadStrip, float> {
  public:
    SiStripBadStripTIDFractionByRun()
        : HistoryPlot<SiStripBadStrip, float>("SiStrip Inner Disks Bad Strip fraction per run",
                                              "TID Bad Strip fraction [%]") {}
    ~SiStripBadStripTIDFractionByRun() override = default;

    float getFromPayload(SiStripBadStrip& payload) override {
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload.getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += payload.decode(*badStrip).range;
        }
        badStripsPerDetId[d] = badStrips;
      }  // loop over detIds

      float numerator(0.), denominator(0.);
      std::vector<uint32_t> all_detids = reader->getAllDetIds();
      for (const auto& det : all_detids) {
        int subid = DetId(det).subdetId();
        if (subid != StripSubdetector::TID)
          continue;
        denominator += 128. * reader->getNumberOfApvsAndStripLength(det).first;
        if (badStripsPerDetId.count(det) != 0)
          numerator += badStripsPerDetId[det];
      }

      delete reader;
      return (numerator / denominator) * 100.;

    }  // payload
  };

  /************************************************
    time history histogram of bad components fraction (TEC)
   *************************************************/

  class SiStripBadStripTECFractionByRun : public HistoryPlot<SiStripBadStrip, float> {
  public:
    SiStripBadStripTECFractionByRun()
        : HistoryPlot<SiStripBadStrip, float>("SiStrip Endcaps Bad Strip fraction per run",
                                              "TEC Bad Strip fraction [%]") {}
    ~SiStripBadStripTECFractionByRun() override = default;

    float getFromPayload(SiStripBadStrip& payload) override {
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::vector<uint32_t> detid;
      payload.getDetIds(detid);

      std::map<uint32_t, int> badStripsPerDetId;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload.getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += payload.decode(*badStrip).range;
        }
        badStripsPerDetId[d] = badStrips;
      }  // loop over detIds

      float numerator(0.), denominator(0.);
      std::vector<uint32_t> all_detids = reader->getAllDetIds();
      for (const auto& det : all_detids) {
        int subid = DetId(det).subdetId();
        if (subid != StripSubdetector::TEC)
          continue;
        denominator += 128. * reader->getNumberOfApvsAndStripLength(det).first;
        if (badStripsPerDetId.count(det) != 0)
          numerator += badStripsPerDetId[det];
      }

      delete reader;
      return (numerator / denominator) * 100.;

    }  // payload
  };

  /************************************************
    Plot BadStrip by region 
  *************************************************/

  class SiStripBadStripByRegion : public PlotImage<SiStripBadStrip, SINGLE_IOV> {
  public:
    SiStripBadStripByRegion()
        : PlotImage<SiStripBadStrip, SINGLE_IOV>("SiStrip BadStrip By Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload(std::get<1>(iov));

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryBadStrips{&m_trackerTopo};
      int totalBadStrips = 0;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = payload->getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += payload->decode(*badStrip).range;
        }
        totalBadStrips += badStrips;
        summaryBadStrips.add(d, badStrips);
      }
      std::map<unsigned int, SiStripDetSummary::Values> mapBadStrips = summaryBadStrips.getCounts();

      //=========================

      TCanvas canvas("BadStrip Region summary", "SiStripBadStrip region summary", 1200, 1000);
      canvas.cd();
      auto h_BadStrips = std::make_unique<TH1F>("BadStripsbyRegion",
                                                "SiStrip Bad Strip summary by region;; n. bad strips",
                                                mapBadStrips.size(),
                                                0.,
                                                mapBadStrips.size());
      h_BadStrips->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto& element : mapBadStrips) {
        iBin++;
        int countBadStrips = (element.second.mean);

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

        h_BadStrips->SetBinContent(iBin, countBadStrips);
        h_BadStrips->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        h_BadStrips->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      h_BadStrips->SetMarkerStyle(21);
      h_BadStrips->SetMarkerSize(1);
      h_BadStrips->SetLineColor(kBlue);
      h_BadStrips->SetLineStyle(9);
      h_BadStrips->SetMarkerColor(kBlue);
      h_BadStrips->GetYaxis()->SetRangeUser(0., h_BadStrips->GetMaximum() * 1.30);
      h_BadStrips->GetYaxis()->SetTitleOffset(1.7);
      h_BadStrips->Draw("HISTsame");
      h_BadStrips->Draw("TEXTsame");

      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(h_BadStrips->GetBinLowEdge(line),
                     canvas.cd()->GetUymin(),
                     h_BadStrips->GetBinLowEdge(line),
                     canvas.cd()->GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
      legend.AddEntry(
          h_BadStrips.get(),
          ("IOV: " + std::to_string(std::get<0>(iov)) + "| n. of bad strips:" + std::to_string(totalBadStrips)).c_str(),
          "PL");
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
    Plot BadStrip by region comparison
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripBadStripByRegionComparisonBase : public PlotImage<SiStripBadStrip, nIOVs, ntags> {
  public:
    SiStripBadStripByRegionComparisonBase()
        : PlotImage<SiStripBadStrip, nIOVs, ntags>("SiStrip BadStrip By Region Comparison"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripBadStrip> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripBadStrip> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      // last payload
      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      SiStripDetSummary summaryLastBadStrips{&m_trackerTopo};
      int totalLastBadStrips = 0;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = last_payload->getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += last_payload->decode(*badStrip).range;
        }
        totalLastBadStrips += badStrips;
        summaryLastBadStrips.add(d, badStrips);
      }
      std::map<unsigned int, SiStripDetSummary::Values> mapLastBadStrips = summaryLastBadStrips.getCounts();

      // first payload
      // needs to be cleared to avoid bias using only detIds of last payload
      detid.clear();
      first_payload->getDetIds(detid);

      SiStripDetSummary summaryFirstBadStrips{&m_trackerTopo};
      int totalFirstBadStrips = 0;

      for (const auto& d : detid) {
        SiStripBadStrip::Range range = first_payload->getRange(d);
        int badStrips(0);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          badStrips += first_payload->decode(*badStrip).range;
        }
        totalFirstBadStrips += badStrips;
        summaryFirstBadStrips.add(d, badStrips);
      }
      std::map<unsigned int, SiStripDetSummary::Values> mapFirstBadStrips = summaryFirstBadStrips.getCounts();

      //=========================

      TCanvas canvas("BadStrip Partion summary", "SiStripBadStrip region summary", 1200, 1000);
      canvas.cd();

      auto h_LastBadStrips = std::make_unique<TH1F>("BadStripsbyRegion1",
                                                    "SiStrip Bad Strip summary by region;; n. bad strips",
                                                    mapLastBadStrips.size(),
                                                    0.,
                                                    mapLastBadStrips.size());
      h_LastBadStrips->SetStats(false);

      auto h_FirstBadStrips = std::make_unique<TH1F>("BadStripsbyRegion2",
                                                     "SiStrip Bad Strip summary by region;; n. bad strips",
                                                     mapFirstBadStrips.size(),
                                                     0.,
                                                     mapFirstBadStrips.size());
      h_FirstBadStrips->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto& element : mapLastBadStrips) {
        iBin++;
        int countBadStrips = (element.second.mean);

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

        h_LastBadStrips->SetBinContent(iBin, countBadStrips);
        h_LastBadStrips->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        h_LastBadStrips->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      // reset the count
      iBin = 0;

      for (const auto& element : mapFirstBadStrips) {
        iBin++;
        int countBadStrips = (element.second.mean);

        h_FirstBadStrips->SetBinContent(iBin, countBadStrips);
        h_FirstBadStrips->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        h_FirstBadStrips->GetXaxis()->LabelsOption("v");
      }

      auto extrema = SiStripPI::getExtrema(h_FirstBadStrips.get(), h_LastBadStrips.get());
      h_LastBadStrips->GetYaxis()->SetRangeUser(extrema.first, extrema.second);

      h_LastBadStrips->SetMarkerStyle(21);
      h_LastBadStrips->SetMarkerSize(1);
      h_LastBadStrips->SetLineColor(kBlue);
      h_LastBadStrips->SetFillColor(kBlue);
      h_LastBadStrips->SetLineStyle(9);
      h_LastBadStrips->SetMarkerColor(kBlue);
      h_LastBadStrips->GetYaxis()->SetRangeUser(0., h_LastBadStrips->GetMaximum() * 1.30);
      h_LastBadStrips->GetYaxis()->SetTitleOffset(1.7);

      h_LastBadStrips->SetBarWidth(0.45);
      h_LastBadStrips->SetBarOffset(0.1);
      h_LastBadStrips->Draw("bar2");
      h_LastBadStrips->Draw("TEXTsame");

      h_FirstBadStrips->SetMarkerStyle(20);
      h_FirstBadStrips->SetMarkerSize(1);
      h_FirstBadStrips->SetFillColor(kRed);
      h_FirstBadStrips->SetLineColor(kRed);
      h_FirstBadStrips->SetLineStyle(1);
      h_FirstBadStrips->SetMarkerColor(kRed);
      h_FirstBadStrips->GetYaxis()->SetTitleOffset(1.7);

      h_FirstBadStrips->SetBarWidth(0.4);
      h_FirstBadStrips->SetBarOffset(0.55);

      h_FirstBadStrips->Draw("bar2same");
      h_FirstBadStrips->Draw("TEXT45same");

      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(h_LastBadStrips->GetBinLowEdge(line),
                     canvas.cd()->GetUymin(),
                     h_LastBadStrips->GetBinLowEdge(line),
                     canvas.cd()->GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(kMagenta);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader("Bad Components comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(
          h_LastBadStrips.get(),
          ("IOV: " + std::to_string(std::get<0>(lastiov)) + "| n. of bad strips:" + std::to_string(totalLastBadStrips))
              .c_str(),
          "PL");
      legend.AddEntry(h_FirstBadStrips.get(),
                      ("IOV: " + std::to_string(std::get<0>(firstiov)) +
                       "| n. of bad strips:" + std::to_string(totalFirstBadStrips))
                          .c_str(),
                      "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  using SiStripBadStripByRegionComparisonSingleTag = SiStripBadStripByRegionComparisonBase<1, MULTI_IOV>;
  using SiStripBadStripByRegionComparisonTwoTags = SiStripBadStripByRegionComparisonBase<2, SINGLE_IOV>;

  /************************************************
    TrackerMap of SiStripBadStrip (bad strips fraction difference)
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripBadStripFractionComparisonTrackerMapBase : public PlotImage<SiStripBadStrip, nIOVs, ntags> {
  public:
    SiStripBadStripFractionComparisonTrackerMapBase()
        : PlotImage<SiStripBadStrip, nIOVs, ntags>("Tracker Map of SiStrip bad strip fraction difference") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripBadStrip> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripBadStrip> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::string titleMap =
          "#Delta fraction of bad Strips per module (IOV:" + lastIOVsince + " - IOV:" + firstIOVsince + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripBadStrips");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid1;
      last_payload->getDetIds(detid1);

      std::map<uint32_t, float> FirstFractionPerDetId;
      std::map<uint32_t, float> LastFractionPerDetId;

      for (const auto& d : detid1) {
        SiStripBadStrip::Range range = last_payload->getRange(d);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          LastFractionPerDetId[d] += last_payload->decode(*badStrip).range;
        }
        // normalize to the number of strips per module
        LastFractionPerDetId[d] /= (128. * reader->getNumberOfApvsAndStripLength(d).first);
      }  // loop over detIds

      std::vector<uint32_t> detid2;
      first_payload->getDetIds(detid2);

      //std::cout << "Size 2: " << detid1.size() << "| Size 1: "<< detid2.size() << std::endl;

      for (const auto& d : detid2) {
        SiStripBadStrip::Range range = first_payload->getRange(d);
        for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
          FirstFractionPerDetId[d] += first_payload->decode(*badStrip).range;
        }
        // normalize to the number of strips per module
        FirstFractionPerDetId[d] /= (128. * reader->getNumberOfApvsAndStripLength(d).first);
      }  // loop over detIds

      std::vector<uint32_t> allDetIds = reader->getAllDetIds();

      int countLastButNotFirst(0);
      int countFirstButNotLast(0);
      int countBoth(0);

      for (const auto& d : allDetIds) {
        if (LastFractionPerDetId.find(d) != LastFractionPerDetId.end() &&
            FirstFractionPerDetId.find(d) == FirstFractionPerDetId.end()) {
          tmap->fill(d, LastFractionPerDetId[d]);
          countLastButNotFirst++;
        } else if (LastFractionPerDetId.find(d) == LastFractionPerDetId.end() &&
                   FirstFractionPerDetId.find(d) != FirstFractionPerDetId.end()) {
          tmap->fill(d, -FirstFractionPerDetId[d]);
          countFirstButNotLast++;
        } else if (LastFractionPerDetId.find(d) != LastFractionPerDetId.end() &&
                   FirstFractionPerDetId.find(d) != FirstFractionPerDetId.end()) {
          float delta = (LastFractionPerDetId[d] - FirstFractionPerDetId[d]);
          if (delta != 0.) {
            tmap->fill(d, delta);
          }
          countBoth++;
        }
      }

#ifdef MMDEBUG
      std::cout << "In 2 but not in 1:" << countLastButNotFirst << std::endl;
      std::cout << "In 1 but not in 2:" << countFirstButNotLast << std::endl;
      std::cout << "In both:" << countBoth << std::endl;
#endif

      //=========================

      std::string fileName(this->m_imageFileName);
      tmap->save(true, 0, 0, fileName);

      delete reader;
      return true;
    }
  };

  using SiStripBadStripFractionComparisonTrackerMapSingleTag =
      SiStripBadStripFractionComparisonTrackerMapBase<1, MULTI_IOV>;
  using SiStripBadStripFractionComparisonTrackerMapTwoTags =
      SiStripBadStripFractionComparisonTrackerMapBase<2, SINGLE_IOV>;

  /************************************************
    Plot BadStrip Quality analysis 
  *************************************************/

  class SiStripBadStripQualityAnalysis : public PlotImage<SiStripBadStrip, SINGLE_IOV> {
  public:
    SiStripBadStripQualityAnalysis()
        : PlotImage<SiStripBadStrip, SINGLE_IOV>("SiStrip BadStrip Quality Analysis"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload(std::get<1>(iov));

      SiStripQuality* siStripQuality_ = new SiStripQuality();
      siStripQuality_->add(payload.get());
      siStripQuality_->cleanUp();
      siStripQuality_->fillBadComponents();

      // store global info

      //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
      int NTkBadComponent[4] = {0};

      //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
      //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
      //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
      int NBadComponent[4][19][4] = {{{0}}};

      // call the filler
      SiStripPI::fillBCArrays(siStripQuality_, NTkBadComponent, NBadComponent, m_trackerTopo);

      //&&&&&&&&&&&&&&&&&&
      // printout
      //&&&&&&&&&&&&&&&&&&

      std::stringstream ss;
      ss.str("");
      ss << "\n-----------------\nGlobal Info\n-----------------";
      ss << "\nBadComponent \t   Modules \tFibers "
            "\tApvs\tStrips\n----------------------------------------------------------------";
      ss << "\nTracker:\t\t" << NTkBadComponent[0] << "\t" << NTkBadComponent[1] << "\t" << NTkBadComponent[2] << "\t"
         << NTkBadComponent[3];
      ss << "\n";
      ss << "\nTIB:\t\t\t" << NBadComponent[0][0][0] << "\t" << NBadComponent[0][0][1] << "\t" << NBadComponent[0][0][2]
         << "\t" << NBadComponent[0][0][3];
      ss << "\nTID:\t\t\t" << NBadComponent[1][0][0] << "\t" << NBadComponent[1][0][1] << "\t" << NBadComponent[1][0][2]
         << "\t" << NBadComponent[1][0][3];
      ss << "\nTOB:\t\t\t" << NBadComponent[2][0][0] << "\t" << NBadComponent[2][0][1] << "\t" << NBadComponent[2][0][2]
         << "\t" << NBadComponent[2][0][3];
      ss << "\nTEC:\t\t\t" << NBadComponent[3][0][0] << "\t" << NBadComponent[3][0][1] << "\t" << NBadComponent[3][0][2]
         << "\t" << NBadComponent[3][0][3];
      ss << "\n";

      for (int i = 1; i < 5; ++i)
        ss << "\nTIB Layer " << i << " :\t\t" << NBadComponent[0][i][0] << "\t" << NBadComponent[0][i][1] << "\t"
           << NBadComponent[0][i][2] << "\t" << NBadComponent[0][i][3];
      ss << "\n";
      for (int i = 1; i < 4; ++i)
        ss << "\nTID+ Disk " << i << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
           << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
      for (int i = 4; i < 7; ++i)
        ss << "\nTID- Disk " << i - 3 << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
           << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
      ss << "\n";
      for (int i = 1; i < 7; ++i)
        ss << "\nTOB Layer " << i << " :\t\t" << NBadComponent[2][i][0] << "\t" << NBadComponent[2][i][1] << "\t"
           << NBadComponent[2][i][2] << "\t" << NBadComponent[2][i][3];
      ss << "\n";
      for (int i = 1; i < 10; ++i)
        ss << "\nTEC+ Disk " << i << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
           << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
      for (int i = 10; i < 19; ++i)
        ss << "\nTEC- Disk " << i - 9 << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
           << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
      ss << "\n";

      edm::LogInfo("SiStripBadStrip_PayloadInspector") << ss.str() << std::endl;
      //std::cout<<  ss.str() << std::endl;

      auto masterTable = std::make_unique<TH2I>("table", "", 4, 0., 4., 39, 0., 39.);

      std::string labelsX[4] = {"Bad Modules", "Bad Fibers", "Bad APVs", "Bad Strips"};
      std::string labelsY[40] = {
          "Tracker",     "TIB",         "TID",         "TOB",         "TEC",         "TIB Layer 1", "TIB Layer 2",
          "TIB Layer 3", "TIB Layer 4", "TID+ Disk 1", "TID+ Disk 2", "TID+ Disk 3", "TID- Disk 1", "TID- Disk 2",
          "TID- Disk 3", "TOB Layer 1", "TOB Layer 2", "TOB Layer 3", "TOB Layer 4", "TOB Layer 5", "TOB Layer 6",
          "TEC+ Disk 1", "TEC+ Disk 2", "TEC+ Disk 3", "TEC+ Disk 4", "TEC+ Disk 5", "TEC+ Disk 6", "TEC+ Disk 7",
          "TEC+ Disk 8", "TEC+ Disk 9", "TEC- Disk 1", "TEC- Disk 2", "TEC- Disk 3", "TEC- Disk 4", "TEC- Disk 5",
          "TEC- Disk 6", "TEC- Disk 7", "TEC- Disk 8", "TEC- Disk 9"};

      for (int iX = 0; iX <= 3; iX++) {
        masterTable->GetXaxis()->SetBinLabel(iX + 1, labelsX[iX].c_str());
      }

      for (int iY = 39; iY >= 1; iY--) {
        masterTable->GetYaxis()->SetBinLabel(iY, labelsY[39 - iY].c_str());
      }

      //                        0 1 2  3
      int layerBoundaries[4] = {4, 6, 6, 18};
      std::vector<int> boundaries;
      boundaries.push_back(39);
      boundaries.push_back(35);

      int cursor = 0;
      int layerIndex = 0;
      for (int iY = 39; iY >= 1; iY--) {
        for (int iX = 0; iX <= 3; iX++) {
          if (iY == 39) {
            masterTable->SetBinContent(iX + 1, iY, NTkBadComponent[iX]);
          } else if (iY >= 35) {
            masterTable->SetBinContent(iX + 1, iY, NBadComponent[(39 - iY) - 1][0][iX]);
          } else {
            if (iX == 0)
              layerIndex++;
            //std::cout<<"iY:"<<iY << " cursor: "  <<cursor << " layerIndex: " << layerIndex << " layer check: "<< layerBoundaries[cursor] <<std::endl;
            masterTable->SetBinContent(iX + 1, iY, NBadComponent[cursor][layerIndex][iX]);
          }
        }
        if (layerIndex == layerBoundaries[cursor]) {
          // bring on the subdet counter and reset the layer count
          cursor++;
          layerIndex = 0;
          boundaries.push_back(iY);
        }
      }

      TCanvas canv("canv", "canv", 800, 800);
      canv.cd();

      canv.SetTopMargin(0.05);
      canv.SetBottomMargin(0.07);
      canv.SetLeftMargin(0.18);
      canv.SetRightMargin(0.05);

      masterTable->GetYaxis()->SetLabelSize(0.04);
      masterTable->GetXaxis()->SetLabelSize(0.05);

      masterTable->SetStats(false);
      canv.SetGrid();

      masterTable->Draw("text");

      canv.Update();
      canv.cd();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(canv.cd()->GetUxmin(),
                     masterTable->GetYaxis()->GetBinLowEdge(line),
                     canv.cd()->GetUxmax(),
                     masterTable->GetYaxis()->GetBinLowEdge(line));
        l[i].SetLineWidth(2);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(kMagenta);
        l[i].Draw("same");
        i++;
      }

      canv.cd();
      TLatex title;
      title.SetTextSize(0.027);
      title.SetTextColor(kBlue);
      title.DrawLatexNDC(0.12, 0.97, ("IOV: " + std::to_string(std::get<0>(iov)) + "| " + std::get<1>(iov)).c_str());
      std::string fileName(m_imageFileName);
      canv.SaveAs(fileName.c_str());

      delete siStripQuality_;
      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Plot BadStrip Quality Comparison
  *************************************************/

  template <int ntags, IOVMultiplicity nIOVs>
  class SiStripBadStripQualityComparisonBase : public PlotImage<SiStripBadStrip, nIOVs, ntags> {
  public:
    SiStripBadStripQualityComparisonBase()
        : PlotImage<SiStripBadStrip, nIOVs, ntags>("SiStrip BadStrip Quality Comparison Analysis"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      //SiStripPI::setPaletteStyle(SiStripPI::BLUERED);
      gStyle->SetPalette(kTemperatureMap);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripBadStrip> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripBadStrip> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      // store global info

      //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
      int f_NTkBadComponent[4] = {0};
      int l_NTkBadComponent[4] = {0};

      // for the total
      int tot_NTkComponents[4] = {0};

      //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
      //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
      //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
      int f_NBadComponent[4][19][4] = {{{0}}};
      int l_NBadComponent[4][19][4] = {{{0}}};

      // for the total
      int totNComponents[4][19][4] = {{{0}}};

      SiStripQuality* f_siStripQuality_ = new SiStripQuality();
      f_siStripQuality_->add(first_payload.get());
      f_siStripQuality_->cleanUp();
      f_siStripQuality_->fillBadComponents();

      // call the filler
      SiStripPI::fillBCArrays(f_siStripQuality_, f_NTkBadComponent, f_NBadComponent, m_trackerTopo);

      SiStripQuality* l_siStripQuality_ = new SiStripQuality();
      l_siStripQuality_->add(last_payload.get());
      l_siStripQuality_->cleanUp();
      l_siStripQuality_->fillBadComponents();

      // call the filler
      SiStripPI::fillBCArrays(l_siStripQuality_, l_NTkBadComponent, l_NBadComponent, m_trackerTopo);

      // fill the total number of components
      SiStripPI::fillTotalComponents(tot_NTkComponents, totNComponents, m_trackerTopo);

      // debug
      //SiStripPI::printBCDebug(f_NTkBadComponent,f_NBadComponent);
      //SiStripPI::printBCDebug(l_NTkBadComponent,l_NBadComponent);

      //SiStripPI::printBCDebug(tot_NTkComponents,totNComponents);

      // declare histograms
      auto masterTable = std::make_unique<TH2F>("table", "", 4, 0., 4., 39, 0., 39.);
      auto masterTableColor = std::make_unique<TH2F>("colortable", "", 4, 0., 4., 39, 0., 39.);

      std::string labelsX[4] = {"Bad Modules", "Bad Fibers", "Bad APVs", "Bad Strips"};
      std::string labelsY[40] = {
          "Tracker",     "TIB",         "TID",         "TOB",         "TEC",         "TIB Layer 1", "TIB Layer 2",
          "TIB Layer 3", "TIB Layer 4", "TID+ Disk 1", "TID+ Disk 2", "TID+ Disk 3", "TID- Disk 1", "TID- Disk 2",
          "TID- Disk 3", "TOB Layer 1", "TOB Layer 2", "TOB Layer 3", "TOB Layer 4", "TOB Layer 5", "TOB Layer 6",
          "TEC+ Disk 1", "TEC+ Disk 2", "TEC+ Disk 3", "TEC+ Disk 4", "TEC+ Disk 5", "TEC+ Disk 6", "TEC+ Disk 7",
          "TEC+ Disk 8", "TEC+ Disk 9", "TEC- Disk 1", "TEC- Disk 2", "TEC- Disk 3", "TEC- Disk 4", "TEC- Disk 5",
          "TEC- Disk 6", "TEC- Disk 7", "TEC- Disk 8", "TEC- Disk 9"};

      for (int iX = 0; iX <= 3; iX++) {
        masterTable->GetXaxis()->SetBinLabel(iX + 1, labelsX[iX].c_str());
        masterTableColor->GetXaxis()->SetBinLabel(iX + 1, labelsX[iX].c_str());
      }

      for (int iY = 39; iY >= 1; iY--) {
        masterTable->GetYaxis()->SetBinLabel(iY, labelsY[39 - iY].c_str());
        masterTableColor->GetYaxis()->SetBinLabel(iY, labelsY[39 - iY].c_str());
      }

      //                        0 1 2 3
      int layerBoundaries[4] = {4, 6, 6, 18};
      std::vector<int> boundaries;
      boundaries.push_back(39);
      boundaries.push_back(35);

      int cursor = 0;
      int layerIndex = 0;
      for (int iY = 39; iY >= 1; iY--) {
        for (int iX = 0; iX <= 3; iX++) {
          if (iY == 39) {
            masterTable->SetBinContent(iX + 1, iY, l_NTkBadComponent[iX] - f_NTkBadComponent[iX]);
            masterTableColor->SetBinContent(
                iX + 1, iY, 100 * float(l_NTkBadComponent[iX] - f_NTkBadComponent[iX]) / tot_NTkComponents[iX]);
            //std::cout<< (l_NTkBadComponent[iX]-f_NTkBadComponent[iX]) << " " << tot_NTkComponents[iX] << " " << float(l_NTkBadComponent[iX]-f_NTkBadComponent[iX])/tot_NTkComponents[iX] << std::endl;
          } else if (iY >= 35) {
            masterTable->SetBinContent(
                iX + 1, iY, (l_NBadComponent[(39 - iY) - 1][0][iX] - f_NBadComponent[(39 - iY) - 1][0][iX]));
            masterTableColor->SetBinContent(
                iX + 1,
                iY,
                100 * float(l_NBadComponent[(39 - iY) - 1][0][iX] - f_NBadComponent[(39 - iY) - 1][0][iX]) /
                    totNComponents[(39 - iY) - 1][0][iX]);
          } else {
            if (iX == 0)
              layerIndex++;
            //std::cout<<"iY:"<<iY << " cursor: "  <<cursor << " layerIndex: " << layerIndex << " layer check: "<< layerBoundaries[cursor] <<std::endl;
            masterTable->SetBinContent(
                iX + 1, iY, (l_NBadComponent[cursor][layerIndex][iX] - f_NBadComponent[cursor][layerIndex][iX]));
            masterTableColor->SetBinContent(
                iX + 1,
                iY,
                100 * float(l_NBadComponent[cursor][layerIndex][iX] - f_NBadComponent[cursor][layerIndex][iX]) /
                    totNComponents[cursor][layerIndex][iX]);
          }
        }
        if (layerIndex == layerBoundaries[cursor]) {
          // bring on the subdet counter and reset the layer count
          cursor++;
          layerIndex = 0;
          boundaries.push_back(iY);
        }
      }

      TCanvas canv("canv", "canv", 1000, 800);
      canv.cd();

      canv.SetTopMargin(0.05);
      canv.SetBottomMargin(0.07);
      canv.SetLeftMargin(0.13);
      canv.SetRightMargin(0.16);

      masterTable->SetStats(false);
      masterTableColor->SetStats(false);
      canv.SetGrid();

      masterTable->SetMarkerColor(kBlack);
      masterTable->SetMarkerSize(1.5);

      float extremum = std::abs(masterTableColor->GetMaximum()) > std::abs(masterTableColor->GetMinimum())
                           ? std::abs(masterTableColor->GetMaximum())
                           : std::abs(masterTableColor->GetMinimum());
      //masterTableColor->Draw("text");
      masterTableColor->GetZaxis()->SetRangeUser(-extremum, extremum);
      masterTableColor->GetZaxis()->SetTitle("percent change [%]");
      masterTableColor->GetZaxis()->CenterTitle(true);
      masterTableColor->GetZaxis()->SetTitleSize(0.05);

      masterTableColor->GetYaxis()->SetLabelSize(0.04);
      masterTableColor->GetXaxis()->SetLabelSize(0.06);

      masterTable->GetYaxis()->SetLabelSize(0.04);
      masterTable->GetXaxis()->SetLabelSize(0.06);

      masterTableColor->Draw("COLZ");
      masterTable->Draw("textsame");

      canv.Update();
      canv.cd();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(canv.cd()->GetUxmin(),
                     masterTable->GetYaxis()->GetBinLowEdge(line),
                     canv.cd()->GetUxmax(),
                     masterTable->GetYaxis()->GetBinLowEdge(line));
        l[i].SetLineWidth(2);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(kMagenta);
        l[i].Draw("same");
        i++;
      }

      canv.cd();
      TLatex title;
      title.SetTextSize(0.045);
      title.SetTextColor(kBlue);
      title.DrawLatexNDC(
          0.33,
          0.96,
          ("#DeltaIOV: " + std::to_string(std::get<0>(lastiov)) + " - " + std::to_string(std::get<0>(firstiov)))
              .c_str());
      std::string fileName(this->m_imageFileName);
      canv.SaveAs(fileName.c_str());

      delete f_siStripQuality_;
      delete l_siStripQuality_;

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  using SiStripBadStripQualityComparisonSingleTag = SiStripBadStripQualityComparisonBase<1, MULTI_IOV>;
  using SiStripBadStripQualityComparisonTwoTags = SiStripBadStripQualityComparisonBase<2, SINGLE_IOV>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripBadStrip) {
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadModuleTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionTkMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionByRun);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTIBFractionByRun);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTOBFractionByRun);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTIDFractionByRun);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTECFractionByRun);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripByRegionComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionComparisonTrackerMapSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionComparisonTrackerMapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripQualityAnalysis);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripQualityComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripQualityComparisonTwoTags);
}
