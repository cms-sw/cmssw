#ifndef CondCore_SiStripPlugins_SiStripCondObjectRepresent_h
#define CondCore_SiStripPlugins_SiStripCondObjectRepresent_h

// system includes
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fmt/printf.h>

// user includes
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

// ROOT includes
#include "TCanvas.h"
#include "TColor.h"
#include "TGaxis.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TPaveLabel.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TStyle.h"

//functions for correct representation of data in summary and plot
namespace SiStripCondObjectRepresent {

  static const std::map<std::string, int> k_colormap = {
      {"TIB", kRed}, {"TOB", kGreen}, {"TID", kBlack}, {"TEC", kBlue}};
  static const std::map<std::string, int> k_markermap = {
      {"TIB", kFullCircle}, {"TOB", kFullTriangleUp}, {"TID", kFullSquare}, {"TEC", kFullTriangleDown}};

  enum plotType { STANDARD, COMPARISON, DIFF, RATIO, MAP, END_OF_TYPES };
  enum granularity { PERSTRIP, PERAPV, PERMODULE };

  template <class type>
  class SiStripCondDataItem {
  public:
    SiStripCondDataItem() { init(); }

    virtual ~SiStripCondDataItem() = default;

    void fillAll(unsigned int detid, const std::vector<type> &store) {
      m_info[detid] = store;
      m_cached = true;
      return;
    }

    void fillByPushBack(unsigned int detid, const type &value) {
      m_info[detid].push_back(value);
      m_cached = true;
    }

    void divide(unsigned int detid, const std::vector<type> &denominator) {
      if (m_info[detid].size() != denominator.size()) {
        throw cms::Exception("Unaligned Conditions")
            << "data size of numerator mismatched the data size of denominator";
      }

      unsigned int counter = 0;
      for (const auto &den : denominator) {
        m_info[detid].at(counter) /= den;
        counter++;
      }
    }

    void subtract(unsigned int detid, const std::vector<type> &subtractor) {
      if (m_info[detid].size() != subtractor.size()) {
        throw cms::Exception("Unaligned Conditions")
            << "data size of numerator mismatched the data size of denominator";
      }

      unsigned int counter = 0;
      for (const auto &sub : subtractor) {
        m_info[detid].at(counter) -= sub;
        counter++;
      }
    }

    std::vector<type> data(unsigned int detid) { return m_info[detid]; }

    std::pair<std::vector<type>, std::vector<type> > demuxedData(unsigned int detid) {
      if (m_compared) {
        std::vector<type> v1(m_info[detid].begin(), m_info[detid].begin() + m_info[detid].size() / 2);
        std::vector<type> v2(m_info[detid].begin() + m_info[detid].size() / 2, m_info[detid].end());
        assert(v1.size() == v2.size());
        return std::make_pair(v1, v2);
      } else {
        throw cms::Exception("Logic error") << "not being in compared mode, data cannot be demultiplexed";
      }
    }

    void fillMonitor1D(const SiStripPI::OpMode &mymode,
                       SiStripPI::Monitor1D *&mon,
                       SiStripPI::Entry &entry,
                       std::vector<type> &values,
                       const unsigned int prev_det,
                       unsigned int &prev_apv,
                       const unsigned int detid) {
      unsigned int istrip = 0;
      for (const auto &value : values) {
        bool flush = false;
        switch (mymode) {
          case (SiStripPI::APV_BASED):
            flush = (prev_det != 0 && prev_apv != istrip / sistrip::STRIPS_PER_APV);
            break;
          case (SiStripPI::MODULE_BASED):
            flush = (prev_det != 0 && prev_det != detid);
            break;
          case (SiStripPI::STRIP_BASED):
            flush = (istrip != 0);
            break;
        }

        if (flush) {
          mon->Fill(prev_apv, prev_det, entry.mean());
          entry.reset();
        }

        entry.add(value);

        prev_apv = istrip / sistrip::STRIPS_PER_APV;
        istrip++;
      }
    }

    void setGranularity(bool isPerStrip, bool isPerAPV) {
      m_servedPerStrip = isPerStrip;
      m_servedPerAPV = isPerAPV;
    }

    bool isCached() { return m_cached; }

    void setComparedBit() { m_compared = true; }

    std::vector<unsigned int> detIds(bool verbose) {
      std::vector<unsigned int> v;
      for (const auto &element : m_info) {
        if (verbose) {
          std::cout << element.first << "\n";
        }
        v.push_back(element.first);
      }

      return v;
    }

  private:
    std::map<unsigned int, std::vector<type> > m_info;
    bool m_servedPerStrip;
    bool m_servedPerAPV;
    bool m_cached;
    bool m_compared;

    void init() {
      m_servedPerStrip = false;
      m_servedPerAPV = false;
      m_info.clear();
      m_cached = false;
      m_compared = false;
    }
  };

  //used to produce all display objects for payload inspector
  template <class Item, class type>
  class SiStripDataContainer {
  public:
    SiStripDataContainer(const std::shared_ptr<Item> &payload,
                         const SiStripPI::MetaData &metadata,
                         const std::string &tagname)
        : payload_(payload),
          run_(std::get<0>(metadata)),
          hash_(std::get<1>(metadata)),
          tagname_(tagname),
          m_trackerTopo(StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())) {
      payloadType_ = std::string();
      granularity_ = PERSTRIP;
      plotMode_ = STANDARD;
      additionalIOV_ = std::make_tuple(-1, "", "");
    }

    virtual ~SiStripDataContainer() = default;

    ///////////////// public get functions  /////////////////
    const unsigned int run() const { return run_; }
    const unsigned int run2() const { return std::get<0>(additionalIOV_); }
    const std::string &hash() const { return hash_; }
    const std::string &hash2() const { return std::get<1>(additionalIOV_); }
    const SiStripPI::MetaData metaData() const { return std::make_tuple(run_, hash_); }
    const std::string &tagName() const { return tagname_; }
    const std::string &tagName2() const { return std::get<2>(additionalIOV_); }
    const std::string &topoMode() const { return TopoMode_; }
    const std::string &payloadName() const { return payloadType_; }
    const plotType &getPlotType() const { return plotMode_; }
    const bool isMultiTag() { return (tagname_ != this->tagName2() && !(this->tagName2()).empty()); }

    void setPlotType(plotType myType) { plotMode_ = myType; }
    void setPayloadType(std::string myPayloadType) { payloadType_ = myPayloadType; }
    void setGranularity(granularity myGranularity) {
      granularity_ = myGranularity;

      switch (myGranularity) {
        case PERSTRIP:
          SiStripCondData_.setGranularity(true, false);
          break;
        case PERAPV:
          SiStripCondData_.setGranularity(false, true);
          break;
        case PERMODULE:
          SiStripCondData_.setGranularity(false, false);
          break;
        default:
          edm::LogError("LogicError") << "Unknown granularity type: " << myGranularity;
      }
    }

    void setAdditionalIOV(const unsigned int run, const std::string &hash, const std::string &tagname) {
      std::get<0>(additionalIOV_) = run;
      std::get<1>(additionalIOV_) = hash;
      std::get<2>(additionalIOV_) = tagname;
    };

    ////NOTE to be implemented in PayloadInspector classes
    virtual void storeAllValues() {
      throw cms::Exception("Value definition not found")
          << "storeAllValues definition not found for " << payloadName() << "\n;";
    };

    SiStripCondDataItem<type> siStripCondData() { return SiStripCondData_; }

    /***********************************************************************/
    const char *plotDescriptor()
    /***********************************************************************/
    {
      const char *thePlotType = "";
      switch (plotMode_) {
        case STANDARD:
          thePlotType = Form("Display - IOV: %i", run_);
          break;
        case COMPARISON:
          thePlotType = "Display";
          break;
        case DIFF:
          thePlotType = Form("#Delta (%i-%i)", run_, std::get<0>(additionalIOV_));
          break;
        case RATIO:
          thePlotType = Form("Ratio (%i/%i)", run_, std::get<0>(additionalIOV_));
          break;
        case MAP:
          thePlotType = Form("TrackerMap - %s", hash_.c_str());
          break;
        case END_OF_TYPES:
          edm::LogError("LogicError") << "Unknown plot type: " << plotMode_;
          break;
        default:
          edm::LogError("LogicError") << "Unknown plot type: " << plotMode_;
          break;
      }

      return thePlotType;
    }

    // all methods needed for comparison of 2 IOVs

    /***********************************************************************/
    void compare(SiStripDataContainer *dataCont2)
    /***********************************************************************/
    {
      plotMode_ = COMPARISON;
      dataCont2->setPlotType(COMPARISON);
      SiStripCondData_.setComparedBit();

      setAdditionalIOV(dataCont2->run(), dataCont2->hash(), dataCont2->tagName());

      if (!SiStripCondData_.isCached())
        storeAllValues();
      dataCont2->storeAllValues();
      auto SiStripCondData2_ = dataCont2->siStripCondData();

      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        auto entriesToAdd = SiStripCondData2_.data(detId);
        for (const auto &entry : entriesToAdd) {
          SiStripCondData_.fillByPushBack(detId, entry);
        }
      }
    }

    /***********************************************************************/
    void divide(SiStripDataContainer *dataCont2)
    /***********************************************************************/
    {
      plotMode_ = RATIO;
      dataCont2->setPlotType(RATIO);

      setAdditionalIOV(dataCont2->run(), dataCont2->hash(), dataCont2->tagName());

      if (!SiStripCondData_.isCached())
        storeAllValues();
      dataCont2->storeAllValues();
      auto SiStripCondData2_ = dataCont2->siStripCondData();

      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        SiStripCondData_.divide(detId, SiStripCondData2_.data(detId));
      }
    }

    /***********************************************************************/
    void subtract(SiStripDataContainer *dataCont2)
    /***********************************************************************/
    {
      plotMode_ = DIFF;
      dataCont2->setPlotType(DIFF);

      setAdditionalIOV(dataCont2->run(), dataCont2->hash(), dataCont2->tagName());

      if (!SiStripCondData_.isCached())
        storeAllValues();
      dataCont2->storeAllValues();
      auto SiStripCondData2_ = dataCont2->siStripCondData();

      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        SiStripCondData_.subtract(detId, SiStripCondData2_.data(detId));
      }
    }

    /***********************************************************************/
    void printAll()
    /***********************************************************************/
    {
      if (!SiStripCondData_.isCached())
        storeAllValues();
      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        std::cout << detId << ": ";
        auto values = SiStripCondData_.data(detId);
        for (const auto &value : values) {
          std::cout << value << " ";
        }
        std::cout << "\n";
      }
    }

    /***********************************************************************/
    void fillTrackerMap(TrackerMap *&tmap,
                        std::pair<float, float> &range,
                        const SiStripPI::estimator &est,
                        const int nsigmas_of_saturation)
    /***********************************************************************/
    {
      std::string titleMap;
      if (plotMode_ != DIFF && plotMode_ != RATIO) {
        titleMap =
            "Tracker Map of " + payloadType_ + " " + estimatorType(est) + " per module (payload : " + hash_ + ")";
      } else {
        titleMap = "Tracker Map of " + payloadType_ + " " + Form("%s", plotDescriptor()) + " " + estimatorType(est) +
                   " per module";
      }

      tmap = new TrackerMap(payloadType_);
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      // storage of info
      std::map<unsigned int, float> info_per_detid;

      if (!SiStripCondData_.isCached())
        storeAllValues();
      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        auto values = SiStripCondData_.data(detId);

        unsigned int nElements = values.size();
        double mean(0.), rms(0.), min(10000.), max(0.);

        for (const auto &value : values) {
          mean += value;
          rms += value * value;
          if (value < min)
            min = value;
          if (value > max)
            max = value;
        }

        mean /= nElements;
        if ((rms / nElements - mean * mean) > 0.) {
          rms = sqrt(rms / nElements - mean * mean);
        } else {
          rms = 0.;
        }

        switch (est) {
          case SiStripPI::min:
            info_per_detid[detId] = min;
            break;
          case SiStripPI::max:
            info_per_detid[detId] = max;
            break;
          case SiStripPI::mean:
            info_per_detid[detId] = mean;
            break;
          case SiStripPI::rms:
            info_per_detid[detId] = rms;
            break;
          default:
            edm::LogWarning("LogicError") << "Unknown estimator: " << est;
            break;
        }
      }

      // loop on the map
      for (const auto &item : info_per_detid) {
        tmap->fill(item.first, item.second);
      }

      range = SiStripPI::getTheRange(info_per_detid, nsigmas_of_saturation);
    }

    /***********************************************************************/
    void fillValuePlot(TCanvas &canvas, const SiStripPI::OpMode &op_mode_, int nbins, float min, float max)
    /***********************************************************************/
    {
      auto myMode = op_mode_;
      // check the consistency first

      if (granularity_ == PERAPV) {
        switch (op_mode_) {
          case SiStripPI::STRIP_BASED:
            edm::LogError("LogicError") << " Cannot display average per " << opType(op_mode_).c_str()
                                        << " in a conditions served per APV";
            return;
          case SiStripPI::APV_BASED:
            myMode = SiStripPI::STRIP_BASED;
            break;
          default:
            break;
        }
      } else if (granularity_ == PERMODULE) {
        if (op_mode_ == SiStripPI::STRIP_BASED || op_mode_ == SiStripPI::APV_BASED) {
          edm::LogError("LogicError") << " Cannot display average per " << opType(op_mode_).c_str()
                                      << " in a conditions served per module";
          return;
        }
      }

      SiStripPI::Monitor1D *f_mon = nullptr;
      SiStripPI::Monitor1D *l_mon = nullptr;

      f_mon = new SiStripPI::Monitor1D(myMode,
                                       "first",
                                       Form("#LT %s #GT per %s;#LT%s per %s#GT %s;n. %ss",
                                            payloadType_.c_str(),
                                            opType(op_mode_).c_str(),
                                            payloadType_.c_str(),
                                            opType(op_mode_).c_str(),
                                            (units_[payloadType_]).c_str(),
                                            opType(op_mode_).c_str()),
                                       nbins,
                                       min,
                                       max);

      if (plotMode_ == COMPARISON) {
        l_mon = new SiStripPI::Monitor1D(myMode,
                                         "last",
                                         Form("#LT %s #GT per %s;#LT%s per %s#GT %s;n. %ss",
                                              payloadType_.c_str(),
                                              opType(op_mode_).c_str(),
                                              payloadType_.c_str(),
                                              opType(op_mode_).c_str(),
                                              (units_[payloadType_]).c_str(),
                                              opType(op_mode_).c_str()),
                                         nbins,
                                         min,
                                         max);
      }

      // retrieve the data
      if (!SiStripCondData_.isCached())
        storeAllValues();
      auto listOfDetIds = SiStripCondData_.detIds(false);

      unsigned int prev_det = 0;
      unsigned int prev_apv = 0;
      SiStripPI::Entry f_entryContainer;
      SiStripPI::Entry l_entryContainer;

      std::cout << "mode:" << opType(myMode) << " granularity: " << granularity_
                << " listOfDetIds.size(): " << listOfDetIds.size() << std::endl;

      for (const auto &detId : listOfDetIds) {
        if (plotMode_ == COMPARISON) {
          auto values = SiStripCondData_.demuxedData(detId);
          SiStripCondData_.fillMonitor1D(myMode, f_mon, f_entryContainer, values.first, prev_det, prev_apv, detId);
          SiStripCondData_.fillMonitor1D(myMode, l_mon, l_entryContainer, values.second, prev_det, prev_apv, detId);
        } else {
          auto values = SiStripCondData_.data(detId);
          SiStripCondData_.fillMonitor1D(myMode, f_mon, l_entryContainer, values, prev_det, prev_apv, detId);
        }
        prev_det = detId;
      }

      TH1F *h_first = (TH1F *)(f_mon->getHist()).Clone("h_first");
      h_first->SetStats(kFALSE);
      SiStripPI::makeNicePlotStyle(h_first);
      h_first->GetYaxis()->CenterTitle(true);
      h_first->GetXaxis()->CenterTitle(true);
      h_first->SetLineWidth(2);
      h_first->SetLineColor(kBlack);

      //=========================
      canvas.cd();
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      //canvas.Modified();

      TLegend *legend = new TLegend(0.52, 0.82, 0.95, 0.9);
      legend->SetTextSize(0.025);

      if (plotMode_ != COMPARISON) {
        float theMax = h_first->GetMaximum();
        h_first->SetMaximum(theMax * 1.30);
        h_first->Draw();

        legend->AddEntry(h_first, Form("IOV: %i", run_), "L");

      } else {
        TH1F *h_last = (TH1F *)(l_mon->getHist()).Clone("h_last");
        h_last->SetStats(kFALSE);
        SiStripPI::makeNicePlotStyle(h_last);
        h_last->GetYaxis()->CenterTitle(true);
        h_last->GetXaxis()->CenterTitle(true);
        h_last->SetLineWidth(2);
        h_last->SetLineColor(kBlue);

        std::cout << h_first->GetEntries() << " ---- " << h_last->GetEntries() << std::endl;

        float theMax = (h_first->GetMaximum() > h_last->GetMaximum()) ? h_first->GetMaximum() : h_last->GetMaximum();

        h_first->SetMaximum(theMax * 1.30);
        h_last->SetMaximum(theMax * 1.30);

        h_first->Draw();
        h_last->Draw("same");

        legend->SetHeader(Form("%s comparison", payloadType_.c_str()), "C");  // option "C" allows to center the header
        legend->AddEntry(h_first, Form("IOV: %i", run_), "F");
        legend->AddEntry(h_last, Form("IOV: %i", (std::get<0>(additionalIOV_))), "F");
      }

      legend->Draw("same");
    }

    /***********************************************************************/
    void fillSummary(TCanvas &canvas)
    /***********************************************************************/
    {
      std::map<unsigned int, SiStripDetSummary::Values> f_map;
      std::map<unsigned int, SiStripDetSummary::Values> l_map;

      if (!SiStripCondData_.isCached())
        storeAllValues();
      auto listOfDetIds = SiStripCondData_.detIds(false);

      if (plotMode_ == COMPARISON) {
        for (const auto &detId : listOfDetIds) {
          auto values = SiStripCondData_.demuxedData(detId);
          for (const auto &value : values.first) {
            summary.add(detId, value);
          }
        }

        f_map = summary.getCounts();
        summary.clear();

        for (const auto &detId : listOfDetIds) {
          auto values = SiStripCondData_.demuxedData(detId);
          for (const auto &value : values.second) {
            summary.add(detId, value);
          }
        }

        l_map = summary.getCounts();

      } else {
        for (const auto &detId : listOfDetIds) {
          auto values = SiStripCondData_.data(detId);
          for (const auto &value : values) {
            summary.add(detId, value);
          }
        }
        f_map = summary.getCounts();
      }

      if (plotMode_ == COMPARISON) {
        std::cout << "f map size: " << f_map.size() << " l map size:" << l_map.size() << std::endl;
        assert(f_map.size() == l_map.size());
      }
      //=========================

      canvas.cd();
      auto h1 = new TH1F(
          "byRegion1",
          Form("SiStrip %s average by region;; average SiStrip %s", payloadType_.c_str(), payloadType_.c_str()),
          f_map.size(),
          0.,
          f_map.size());
      h1->SetStats(false);

      auto h2 = new TH1F(
          "byRegion2",
          Form("SiStrip %s average by region;; average SiStrip %s", payloadType_.c_str(), payloadType_.c_str()),
          f_map.size(),
          0.,
          f_map.size());
      h2->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.17);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : f_map) {
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

      iBin = 0;
      if (plotMode_ == COMPARISON) {
        for (const auto &element : l_map) {
          iBin++;
          int count = element.second.count;
          double mean = (element.second.mean) / count;

          h2->SetBinContent(iBin, mean);
          h2->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
          h2->GetXaxis()->LabelsOption("v");
        }
      }

      h1->GetYaxis()->SetRangeUser(0., h1->GetMaximum() * 1.30);
      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1.5);
      h1->Draw("HIST");
      h1->Draw("Psame");

      if (plotMode_ == COMPARISON) {
        h2->GetYaxis()->SetRangeUser(0., h2->GetMaximum() * 1.30);
        h2->SetMarkerStyle(25);
        h2->SetMarkerColor(kBlue);
        h2->SetLineColor(kBlue);
        h2->SetMarkerSize(1.5);
        h2->Draw("HISTsame");
        h2->Draw("Psame");
      }

      canvas.Update();

      TLine *l[boundaries.size()];
      unsigned int i = 0;
      for (const auto &line : boundaries) {
        l[i] = new TLine(h1->GetBinLowEdge(line), canvas.GetUymin(), h1->GetBinLowEdge(line), canvas.GetUymax());
        l[i]->SetLineWidth(1);
        l[i]->SetLineStyle(9);
        l[i]->SetLineColor(2);
        l[i]->Draw("same");
        i++;
      }

      TLegend *legend = new TLegend(0.52, 0.82, 0.95, 0.9);
      legend->SetHeader(hash_.c_str(), "C");  // option "C" allows to center the header
      legend->AddEntry(h1, Form("IOV: %i", run_), "PL");
      if (plotMode_ == COMPARISON) {
        legend->AddEntry(h2, Form("IOV: %i", std::get<0>(additionalIOV_)), "PL");
      }
      legend->SetTextSize(0.025);
      legend->Draw("same");
    }

    /***********************************************************************/
    void fillByPartition(TCanvas &canvas, int nbins, float min, float max)
    /***********************************************************************/
    {
      std::map<std::string, TH1F *> h_parts;
      std::map<std::string, TH1F *> h_parts2;
      std::vector<std::string> parts = {"TEC", "TOB", "TIB", "TID"};

      const char *device;
      switch (granularity_) {
        case PERSTRIP:
          device = "strips";
          break;
        case PERAPV:
          device = "APVs";
          break;
        case PERMODULE:
          device = "modules";
          break;
        default:
          device = "unrecognized device";
          break;
      }

      for (const auto &part : parts) {
        TString globalTitle = Form("%s - %s %s;%s %s;n. %s",
                                   plotDescriptor(),
                                   payloadType_.c_str(),
                                   part.c_str(),
                                   payloadType_.c_str(),
                                   (units_[payloadType_]).c_str(),
                                   device);

        h_parts[part] = new TH1F(Form("h_%s", part.c_str()), globalTitle, nbins, min, max);
        h_parts[part]->SetTitle(""); /* remove the title from display */
        if (plotMode_ == COMPARISON) {
          h_parts2[part] = new TH1F(Form("h2_%s", part.c_str()), globalTitle, nbins, min, max);
          h_parts2[part]->SetTitle(""); /* remove the title from display */
        }
      }

      if (!SiStripCondData_.isCached())
        storeAllValues();
      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        auto values = SiStripCondData_.data(detId);
        int subid = DetId(detId).subdetId();
        unsigned int counter{0};
        for (const auto &value : values) {
          counter++;
          switch (subid) {
            case StripSubdetector::TIB:
              if ((plotMode_ == COMPARISON) && (counter > (values.size() / 2))) {
                h_parts2["TIB"]->Fill(value);
              } else {
                h_parts["TIB"]->Fill(value);
              }
              break;
            case StripSubdetector::TID:
              if ((plotMode_ == COMPARISON) && (counter > (values.size() / 2))) {
                h_parts2["TID"]->Fill(value);
              } else {
                h_parts["TID"]->Fill(value);
              }
              break;
            case StripSubdetector::TOB:
              if ((plotMode_ == COMPARISON) && (counter > (values.size() / 2))) {
                h_parts2["TOB"]->Fill(value);
              } else {
                h_parts["TOB"]->Fill(value);
              }
              break;
            case StripSubdetector::TEC:
              if ((plotMode_ == COMPARISON) && (counter > (values.size() / 2))) {
                h_parts2["TEC"]->Fill(value);
              } else {
                h_parts["TEC"]->Fill(value);
              }
              break;
            default:
              edm::LogError("LogicError") << "Unknown partition: " << subid;
              break;
          }
        }
      }

      canvas.Divide(2, 2);

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(31);

      int index = 0;
      for (const auto &part : parts) {
        index++;
        canvas.cd(index)->SetTopMargin(0.07);
        canvas.cd(index)->SetLeftMargin(0.13);
        canvas.cd(index)->SetRightMargin(0.08);

        SiStripPI::makeNicePlotStyle(h_parts[part]);
        h_parts[part]->SetMinimum(1.);
        h_parts[part]->SetStats(false);
        h_parts[part]->SetLineWidth(2);

        if (plotMode_ != COMPARISON) {
          h_parts[part]->SetLineColor(k_colormap.at(part));
          h_parts[part]->SetFillColorAlpha(k_colormap.at(part), 0.15);
          float theMax = h_parts[part]->GetMaximum();
          h_parts[part]->SetMaximum(theMax * 1.30);
        } else {
          h_parts[part]->SetLineColor(kBlack);

          SiStripPI::makeNicePlotStyle(h_parts2[part]);
          h_parts2[part]->SetMinimum(1.);
          h_parts2[part]->SetStats(false);
          h_parts2[part]->SetLineWidth(2);
          h_parts2[part]->SetLineColor(kBlue);
          h_parts2[part]->SetFillColorAlpha(kBlue, 0.15);

          float theMax = (h_parts[part]->GetMaximum() > h_parts2[part]->GetMaximum()) ? h_parts[part]->GetMaximum()
                                                                                      : h_parts2[part]->GetMaximum();

          h_parts[part]->SetMaximum(theMax * 1.30);
          h_parts2[part]->SetMaximum(theMax * 1.30);
        }

        h_parts[part]->Draw();
        if (plotMode_ == COMPARISON) {
          h_parts2[part]->Draw("same");
        }

        TLegend *leg = new TLegend(.13, 0.81, 0.92, 0.93);
        if (plotMode_ != COMPARISON) {
          // it means it's a difference
          if (this->isMultiTag()) {
            leg->SetHeader("#bf{Two Tags Difference}", "C");  // option "C" allows to center the header
            leg->AddEntry(h_parts[part],
                          (fmt::sprintf("#splitline{%s : %i}{%s : %i}", tagName(), run(), tagName2(), run2())).c_str(),
                          "F");
          } else {
            leg->SetHeader(("tag: #bf{" + tagName() + "}").c_str(), "C");  // option "C" allows to center the header
            leg->AddEntry(h_parts[part], (fmt::sprintf("%s", plotDescriptor())).c_str(), "F");
          }
          leg->SetTextSize(0.04);
          leg->Draw("same");
          ltx.DrawLatexNDC(
              0.35,
              0.7,
              (fmt::sprintf("#splitline{#mu = %.2f}{r.m.s. = %.2f}", h_parts[part]->GetMean(), h_parts[part]->GetRMS()))
                  .c_str());
          ltx.DrawLatexNDC(1 - gPad->GetRightMargin(),
                           1 - gPad->GetTopMargin() + 0.01,
                           (fmt::sprintf("#color[2]{%s} %s Values %s", part, payloadType_, plotDescriptor())).c_str());
        } else {
          if (this->isMultiTag()) {
            leg->SetHeader("#bf{Two Tags Comparison}", "C");  // option "C" allows to center the header
            leg->AddEntry(h_parts[part], (fmt::sprintf("%s : %i", tagName(), run())).c_str(), "F");
            leg->AddEntry(h_parts2[part], (fmt::sprintf("%s : %i", tagName2(), run2())).c_str(), "F");
          } else {
            leg->SetHeader(("tag: #bf{" + tagName() + "}").c_str(), "C");  // option "C" allows to center the header
            leg->AddEntry(h_parts[part], (fmt::sprintf("IOV since: %i", this->run())).c_str(), "F");
            leg->AddEntry(h_parts2[part], (fmt::sprintf("IOV since: %i", this->run2())).c_str(), "F");
          }
          leg->SetTextSize(0.035);
          leg->Draw("same");
          ltx.DrawLatexNDC(1 - gPad->GetRightMargin(),
                           1 - gPad->GetTopMargin() + 0.01,
                           (fmt::sprintf("#color[2]{%s} %s Values Comparison", part, payloadType_)).c_str());
        }
      }
    }

    /***********************************************************************/
    void fillCorrelationByPartition(TCanvas &canvas, int nbins, float min, float max)
    /***********************************************************************/
    {
      SiStripPI::setPaletteStyle(SiStripPI::DEFAULT); /* better looking palette ;)*/

      if (plotMode_ != COMPARISON) {
        throw cms::Exception("Logic error") << "not being in compared mode, cannot plot correlations";
      }

      std::map<std::string, TH2F *> h2_parts;
      std::vector<std::string> parts = {"TEC", "TOB", "TIB", "TID"};

      const char *device;
      switch (granularity_) {
        case PERSTRIP:
          device = "strips";
          break;
        case PERAPV:
          device = "APVs";
          break;
        case PERMODULE:
          device = "modules";
          break;
        default:
          device = "unrecognized device";
          break;
      }

      for (const auto &part : parts) {
        TString globalTitle = Form("%s - %s %s;%s %s (#color[4]{%s});%s %s (#color[4]{%s});n. %s",
                                   "Correlation",  // FIXME: can use the plotDescriptor()
                                   payloadType_.c_str(),
                                   part.c_str(),
                                   payloadType_.c_str(),
                                   (units_[payloadType_]).c_str(),
                                   std::to_string(run_).c_str(),
                                   payloadType_.c_str(),
                                   (units_[payloadType_]).c_str(),
                                   std::to_string(std::get<0>(additionalIOV_)).c_str(),
                                   device);

        h2_parts[part] = new TH2F(Form("h2_%s", part.c_str()), globalTitle, nbins, min, max, nbins, min, max);
      }

      if (!SiStripCondData_.isCached())
        storeAllValues();
      auto listOfDetIds = SiStripCondData_.detIds(false);
      for (const auto &detId : listOfDetIds) {
        auto values = SiStripCondData_.demuxedData(detId);
        int subid = DetId(detId).subdetId();
        unsigned int counter{0};
        for (const auto &value : values.first) {
          switch (subid) {
            case StripSubdetector::TIB:
              h2_parts["TIB"]->Fill(value, (values.second)[counter]);
              break;
            case StripSubdetector::TID:
              h2_parts["TID"]->Fill(value, (values.second)[counter]);
              break;
            case StripSubdetector::TOB:
              h2_parts["TOB"]->Fill(value, (values.second)[counter]);
              break;
            case StripSubdetector::TEC:
              h2_parts["TEC"]->Fill(value, (values.second)[counter]);
              break;
            default:
              edm::LogError("LogicError") << "Unknown partition: " << subid;
              break;
          }
          counter++;
        }
      }

      canvas.Divide(2, 2);

      int index = 0;
      for (const auto &part : parts) {
        index++;
        canvas.cd(index)->SetTopMargin(0.07);
        canvas.cd(index)->SetLeftMargin(0.13);
        canvas.cd(index)->SetRightMargin(0.17);

        SiStripPI::makeNicePlotStyle(h2_parts[part]);
        h2_parts[part]->GetZaxis()->SetTitleOffset(1.6);
        h2_parts[part]->GetZaxis()->SetTitleSize(0.04);
        h2_parts[part]->GetZaxis()->CenterTitle();
        h2_parts[part]->GetZaxis()->SetMaxDigits(2); /* exponentiate z-axis */

        //h2_parts[part]->SetMarkerColor(k_colormap.at(part));
        //h2_parts[part]->SetMarkerStyle(k_markermap.at(part));
        //h2_parts[part]->SetStats(false);
        //h2_parts[part]->Draw("P");
        h2_parts[part]->Draw("colz");

        TLegend *leg = new TLegend(.13, 0.87, 0.27, 0.93);
        leg->SetTextSize(0.045);
        leg->SetHeader(Form("#bf{%s}", part.c_str()), "C");  // option "C" allows to center the header
        //leg->AddEntry(h2_parts[part], Form("#DeltaIOV: #splitline{%i}{%i}", run_, std::get<0>(additionalIOV_)),"P");
        leg->Draw("same");
      }
    }

  protected:
    std::shared_ptr<Item> payload_;
    std::string payloadType_;
    SiStripCondDataItem<type> SiStripCondData_;

  private:
    unsigned int run_;
    std::string hash_;
    std::string tagname_;
    granularity granularity_;
    std::string TopoMode_;
    TrackerTopology m_trackerTopo;
    SiStripDetSummary summary{&m_trackerTopo};
    // "Map", "Ratio", or "Diff"
    plotType plotMode_;
    std::tuple<int, std::string, std::string> additionalIOV_;

    std::map<std::string, std::string> units_ = {{"SiStripPedestals", "[ADC counts]"},
                                                 {"SiStripApvGain", ""},  //dimensionless TODO: verify
                                                 {"SiStripNoises", "[ADC counts]"},
                                                 {"SiStripLorentzAngle", "[1/T}]"},
                                                 {"SiStripBackPlaneCorrection", ""},
                                                 {"SiStripBadStrip", ""},  // dimensionless
                                                 {"SiStripDetVOff", ""}};  // dimensionless

    std::string opType(SiStripPI::OpMode mode) {
      std::string types[3] = {"Strip", "APV", "Module"};
      return types[mode];
    }
  };

}  // namespace SiStripCondObjectRepresent

#endif
