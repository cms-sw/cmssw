#ifndef CONDCORE_SIPIXELPLUGINS_SIPIXELGAINCALIBHELPER_H
#define CONDCORE_SIPIXELPLUGINS_SIPIXELGAINCALIBHELPER_H

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondCore/SiPixelPlugins/interface/PixelRegionContainers.h"

#include <type_traits>
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

namespace gainCalibHelper {

  using AvgMap = std::map<uint32_t, float>;

  namespace gainCalibPI {

    enum type { t_gain = 0, t_pedestal = 1, t_correlation = 2 };

    //============================================================================
    // helper method to fill the gain / pedestals distributions
    template <typename PayloadType>
    static void fillTheHisto(const std::shared_ptr<PayloadType>& payload,
                             std::shared_ptr<TH1F> h1,
                             gainCalibPI::type theType,
                             const std::vector<uint32_t>& wantedIds = {}) {
      std::vector<uint32_t> detids;
      if (wantedIds.empty()) {
        payload->getDetIds(detids);
      } else {
        detids.assign(wantedIds.begin(), wantedIds.end());
      }

      for (const auto& d : detids) {
        // skip the special case used to signal there are no attached dets
        if (d == 0xFFFFFFFF)
          continue;

        auto range = payload->getRange(d);
        int numberOfRowsToAverageOver = payload->getNumberOfRowsToAverageOver();
        int ncols = payload->getNCols(d);
        int nRocsInRow = (range.second - range.first) / ncols / numberOfRowsToAverageOver;
        unsigned int nRowsForHLT = 1;
        int nrows = std::max((payload->getNumberOfRowsToAverageOver() * nRocsInRow),
                             nRowsForHLT);  // dirty trick to make it work for the HLT payload

        auto rangeAndCol = payload->getRangeAndNCols(d);
        bool isDeadColumn;
        bool isNoisyColumn;

        COUT << "NCOLS: " << payload->getNCols(d) << " " << rangeAndCol.second << " NROWS:" << nrows
             << ", RANGES: " << rangeAndCol.first.second - rangeAndCol.first.first
             << ", Ratio: " << float(rangeAndCol.first.second - rangeAndCol.first.first) / rangeAndCol.second
             << std::endl;

        float quid(-99999.);

        for (int col = 0; col < ncols; col++) {
          for (int row = 0; row < nrows; row++) {
            switch (theType) {
              case gainCalibPI::t_gain:
                quid = payload->getGain(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);
                break;
              case gainCalibPI::t_pedestal:
                quid = payload->getPed(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);
                break;
              default:
                edm::LogError("gainCalibPI::fillTheHisto") << "Unrecognized type " << theType << std::endl;
                break;
            }
            h1->Fill(quid);
          }  // loop on rows
        }    // loop on cols
      }      // loop on detids
    }        // fillTheHisto

    //============================================================================
    // helper method to fill the gain / pedestal averages per module maps
    template <typename PayloadType>
    static void fillThePerModuleMap(const std::shared_ptr<PayloadType>& payload,
                                    AvgMap& map,
                                    gainCalibPI::type theType) {
      std::vector<uint32_t> detids;
      payload->getDetIds(detids);

      for (const auto& d : detids) {
        auto range = payload->getRange(d);
        int numberOfRowsToAverageOver = payload->getNumberOfRowsToAverageOver();
        int ncols = payload->getNCols(d);
        int nRocsInRow = (range.second - range.first) / ncols / numberOfRowsToAverageOver;
        unsigned int nRowsForHLT = 1;
        int nrows = std::max((payload->getNumberOfRowsToAverageOver() * nRocsInRow),
                             nRowsForHLT);  // dirty trick to make it work for the HLT payload

        auto rangeAndCol = payload->getRangeAndNCols(d);
        bool isDeadColumn;
        bool isNoisyColumn;

        float sumOfX(0.);
        int nPixels(0);
        for (int col = 0; col < ncols; col++) {
          for (int row = 0; row < nrows; row++) {
            nPixels++;
            switch (theType) {
              case gainCalibPI::t_gain:
                sumOfX +=
                    payload->getGain(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);
                break;
              case gainCalibPI::t_pedestal:
                sumOfX += payload->getPed(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);
                break;
              default:
                edm::LogError("gainCalibPI::fillThePerModuleMap") << "Unrecognized type " << theType << std::endl;
                break;
            }  // switch on the type
          }    // rows
        }      // columns
        // fill the return value map
        map[d] = sumOfX / nPixels;
      }  // loop on the detId
    }    // fillThePerModuleMap

    //============================================================================
    // helper method to fill the gain / pedestals distributions
    template <typename PayloadType>
    static void fillTheHistos(const std::shared_ptr<PayloadType>& payload,
                              std::shared_ptr<TH1> hBPix,
                              std::shared_ptr<TH1> hFPix,
                              gainCalibPI::type theType) {
      std::vector<uint32_t> detids;
      payload->getDetIds(detids);

      bool isCorrelation_ = hBPix.get()->InheritsFrom(TH2::Class()) && (theType == gainCalibPI::t_correlation);

      for (const auto& d : detids) {
        int subid = DetId(d).subdetId();
        auto range = payload->getRange(d);
        int numberOfRowsToAverageOver = payload->getNumberOfRowsToAverageOver();
        int ncols = payload->getNCols(d);
        int nRocsInRow = (range.second - range.first) / ncols / numberOfRowsToAverageOver;
        unsigned int nRowsForHLT = 1;
        int nrows = std::max((payload->getNumberOfRowsToAverageOver() * nRocsInRow),
                             nRowsForHLT);  // dirty trick to make it work for the HLT payload

        auto rangeAndCol = payload->getRangeAndNCols(d);
        bool isDeadColumn;
        bool isNoisyColumn;

        for (int col = 0; col < ncols; col++) {
          for (int row = 0; row < nrows; row++) {
            float gain = payload->getGain(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);
            float ped = payload->getPed(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);

            switch (subid) {
              case PixelSubdetector::PixelBarrel: {
                if (isCorrelation_) {
                  hBPix->Fill(gain, ped);
                } else {
                  hBPix->Fill((theType == gainCalibPI::t_gain ? gain : ped));
                }
                break;
              }
              case PixelSubdetector::PixelEndcap: {
                if (isCorrelation_) {
                  hFPix->Fill(gain, ped);
                } else {
                  hFPix->Fill((theType == gainCalibPI::t_gain ? gain : ped));
                }
                break;
              }
              default:
                edm::LogError("gainCalibPI::fillTheHistos") << d << " is not a Pixel DetId" << std::endl;
                break;
            }  // switch on subid
          }    // row loop
        }      // column loop
      }        // detid loop
    }          // filltheHistos
  }            // namespace gainCalibPI

  constexpr char const* TypeName[2] = {"Gains", "Pedestals"};

  /*******************************************************************
    1d histogram of SiPixelGainCalibration for Gains of 1 IOV 
  ********************************************************************/
  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationValues
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGainCalibrationValues()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              Form("SiPixelGainCalibration %s Values", TypeName[myType])) {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      gStyle->SetOptStat("emr");

      float minimum(9999.);
      float maximum(-9999.);

      switch (myType) {
        case gainCalibPI::t_gain:
          maximum = payload->getGainHigh();
          minimum = payload->getGainLow();
          break;
        case gainCalibPI::t_pedestal:
          maximum = payload->getPedHigh();
          minimum = payload->getPedLow();
          break;
        default:
          edm::LogError(label_) << "Unrecognized type " << myType << std::endl;
          break;
      }

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      auto h1 = std::make_shared<TH1F>(Form("%s values", TypeName[myType]),
                                       Form("SiPixel Gain Calibration %s - %s;per %s %s;# %ss",
                                            (isForHLT_ ? "ForHLT" : "Offline"),
                                            TypeName[myType],
                                            (isForHLT_ ? "Column" : "Pixel"),
                                            TypeName[myType],
                                            (isForHLT_ ? "column" : "pixel")),
                                       200,
                                       minimum,
                                       maximum);
      canvas.cd();
      SiPixelPI::adjustCanvasMargins(canvas.cd(), 0.06, 0.12, 0.12, 0.05);
      canvas.Modified();

      // fill the histogram
      gainCalibPI::fillTheHisto(payload, h1, myType);

      canvas.cd()->SetLogy();
      h1->SetTitle("");
      h1->GetYaxis()->SetRangeUser(0.1, h1->GetMaximum() * 10.);
      h1->SetFillColor(kBlue);
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
      legend.SetLineColor(10);
      legend.SetTextSize(0.025);
      legend.Draw("same");

      TPaveStats* st = (TPaveStats*)h1->FindObject("stats");
      st->SetTextSize(0.03);
      SiPixelPI::adjustStats(st, 0.15, 0.83, 0.39, 0.93);

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.1,
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Gain Calibration IOV:" + std::to_string(std::get<0>(iov))).c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /*******************************************************************
    1d histograms per region of SiPixelGainCalibration for Gains of 1 IOV
  ********************************************************************/
  template <bool isBarrel, gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationValuesPerRegion
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGainCalibrationValuesPerRegion()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              Form("SiPixelGainCalibration %s Values Per Region", TypeName[myType])) {
      cond::payloadInspector::PlotBase::addInputParam("SetLog");

      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetOptStat("mr");

      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      // parse first if log
      bool setLog(true);
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("SetLog");
      if (ip != paramValues.end()) {
        auto answer = boost::lexical_cast<std::string>(ip->second);
        if (!SiPixelPI::checkAnswerOK(answer, setLog)) {
          throw cms::Exception(label_)
              << "\nERROR: " << answer
              << " is not a valid setting for this parameter, please use True,False,1,0,Yes,No \n\n";
        }
      }

      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      std::vector<uint32_t> detids;
      payload->getDetIds(detids);

      float minimum(9999.);
      float maximum(-9999.);

      switch (myType) {
        case gainCalibPI::t_gain:
          maximum = payload->getGainHigh();
          minimum = payload->getGainLow();
          break;
        case gainCalibPI::t_pedestal:
          maximum = payload->getPedHigh();
          minimum = payload->getPedLow();
          break;
        default:
          edm::LogError(label_) << "Unrecognized type " << myType << std::endl;
          break;
      }

      TCanvas canvas("Canv", "Canv", isBarrel ? 1400 : 1800, 1200);
      if (detids.size() > SiPixelPI::phase1size) {
        SiPixelPI::displayNotSupported(canvas, detids.size());
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      canvas.cd();

      SiPixelPI::PhaseInfo phaseInfo(detids.size());
      const char* path_toTopologyXML = phaseInfo.pathToTopoXML();

      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto myPlots = PixelRegions::PixelRegionContainers(&tTopo, phaseInfo.phase());
      myPlots.bookAll(Form("SiPixel Gain Calibration %s - %s", (isForHLT_ ? "ForHLT" : "Offline"), TypeName[myType]),
                      Form("per %s %s", (isForHLT_ ? "Column" : "Pixel"), TypeName[myType]),
                      Form("# %ss", (isForHLT_ ? "column" : "pixel")),
                      200,
                      minimum,
                      maximum);

      canvas.Modified();

      // fill the histograms
      for (const auto& pixelId : PixelRegions::PixelIDs) {
        auto wantedDets = PixelRegions::attachedDets(pixelId, &tTopo, phaseInfo.phase());
        gainCalibPI::fillTheHisto(payload, myPlots.getHistoFromMap(pixelId), myType, wantedDets);
      }

      if (setLog) {
        myPlots.setLogScale();
      }
      myPlots.beautify(kBlue, -1);
      myPlots.draw(canvas, isBarrel, "HIST");

      TLegend legend = TLegend(0.45, 0.88, 0.91, 0.92);
      legend.SetHeader(("hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      //legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetLineColor(10);
      legend.SetTextSize(0.025);
      legend.Draw("same");

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
        auto leftX = setLog ? 0. : 0.1;
        ltx.DrawLatexNDC(gPad->GetLeftMargin() + leftX,
                         1 - gPad->GetTopMargin() + 0.01,
                         (PixelRegions::IDlabels.at(index) + ", IOV:" + std::to_string(std::get<0>(iov))).c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /*******************************************************************
    1d histograms comparison per region of SiPixelGainCalibration for Gains of 2 IOV
  ********************************************************************/
  template <bool isBarrel,
            gainCalibPI::type myType,
            cond::payloadInspector::IOVMultiplicity nIOVs,
            int ntags,
            class PayloadType>
  class SiPixelGainCalibrationValuesComparisonPerRegion
      : public cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags> {
  public:
    SiPixelGainCalibrationValuesComparisonPerRegion()
        : cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags>(
              Form("SiPixelGainCalibration %s Values Per Region %i tag(s)", TypeName[myType], ntags)) {
      cond::payloadInspector::PlotBase::addInputParam("SetLog");

      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetOptStat("mr");

      COUT << "ntags: " << ntags << " this->m_plotAnnotations.ntags: " << this->m_plotAnnotations.ntags << std::endl;

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      // parse first if log
      bool setLog(true);
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("SetLog");
      if (ip != paramValues.end()) {
        auto answer = boost::lexical_cast<std::string>(ip->second);
        if (!SiPixelPI::checkAnswerOK(answer, setLog)) {
          throw cms::Exception(label_)
              << "\nERROR: " << answer
              << " is not a valid setting for this parameter, please use True,False,1,0,Yes,No \n\n";
        }
      }

      std::shared_ptr<PayloadType> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<PayloadType> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<uint32_t> f_detids, l_detids;
      last_payload->getDetIds(l_detids);
      first_payload->getDetIds(f_detids);

      float minimum(9999.);
      float maximum(-9999.);

      switch (myType) {
        case gainCalibPI::t_gain:
          maximum = std::max(last_payload->getGainHigh(), first_payload->getGainHigh());
          minimum = std::min(last_payload->getGainLow(), first_payload->getGainLow());
          break;
        case gainCalibPI::t_pedestal:
          maximum = std::max(last_payload->getPedHigh(), first_payload->getPedHigh());
          minimum = std::min(last_payload->getPedLow(), first_payload->getPedLow());
          break;
        default:
          edm::LogError(label_) << "Unrecognized type " << myType << std::endl;
          break;
      }

      TCanvas canvas("Canv", "Canv", isBarrel ? 1400 : 1800, 1200);
      if (std::max(l_detids.size(), f_detids.size()) > SiPixelPI::phase1size) {
        SiPixelPI::displayNotSupported(canvas, std::max(f_detids.size(), l_detids.size()));
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      canvas.cd();

      SiPixelPI::PhaseInfo l_phaseInfo(l_detids.size());
      SiPixelPI::PhaseInfo f_phaseInfo(f_detids.size());
      const char* path_toTopologyXML = l_phaseInfo.pathToTopoXML();

      auto l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto l_myPlots = PixelRegions::PixelRegionContainers(&l_tTopo, l_phaseInfo.phase());
      l_myPlots.bookAll(
          Form("Last SiPixel Gain Calibration %s - %s", (isForHLT_ ? "ForHLT" : "Offline"), TypeName[myType]),
          Form("per %s %s", (isForHLT_ ? "Column" : "Pixel"), TypeName[myType]),
          Form("# %ss", (isForHLT_ ? "column" : "pixel")),
          200,
          minimum,
          maximum);

      path_toTopologyXML = f_phaseInfo.pathToTopoXML();

      auto f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto f_myPlots = PixelRegions::PixelRegionContainers(&f_tTopo, f_phaseInfo.phase());
      f_myPlots.bookAll(
          Form("First SiPixel Gain Calibration %s - %s", (isForHLT_ ? "ForHLT" : "Offline"), TypeName[myType]),
          Form("per %s %s", (isForHLT_ ? "Column" : "Pixel"), TypeName[myType]),
          Form("# %ss", (isForHLT_ ? "column" : "pixel")),
          200,
          minimum,
          maximum);

      // fill the histograms
      for (const auto& pixelId : PixelRegions::PixelIDs) {
        auto f_wantedDets = PixelRegions::attachedDets(pixelId, &f_tTopo, f_phaseInfo.phase());
        auto l_wantedDets = PixelRegions::attachedDets(pixelId, &l_tTopo, l_phaseInfo.phase());
        gainCalibPI::fillTheHisto(first_payload, f_myPlots.getHistoFromMap(pixelId), myType, f_wantedDets);
        gainCalibPI::fillTheHisto(last_payload, l_myPlots.getHistoFromMap(pixelId), myType, l_wantedDets);
      }

      if (setLog) {
        f_myPlots.setLogScale();
        l_myPlots.setLogScale();
      }

      l_myPlots.beautify(kRed, -1);
      f_myPlots.beautify(kAzure, -1);

      l_myPlots.draw(canvas, isBarrel, "HIST", f_phaseInfo.isPhase1Comparison(l_phaseInfo));
      f_myPlots.draw(canvas, isBarrel, "HISTsames", f_phaseInfo.isPhase1Comparison(l_phaseInfo));

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
        auto leftX = setLog ? 0. : 0.1;
        ltx.DrawLatexNDC(gPad->GetLeftMargin() + leftX,
                         1 - gPad->GetTopMargin() + 0.01,
                         (PixelRegions::IDlabels.at(index) + " : #color[4]{" + std::to_string(std::get<0>(firstiov)) +
                          "} vs #color[2]{" + std::to_string(std::get<0>(lastiov)) + "}")
                             .c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /*******************************************************************
    1d histogram of SiPixelGainCalibration for Gain/Pedestals
    correlation of 1 IOV
  ********************************************************************/
  template <class PayloadType>
  class SiPixelGainCalibrationCorrelations
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGainCalibrationCorrelations()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelGainCalibration gain/pedestal correlations") {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      gStyle->SetOptStat("emr");
      gStyle->SetPalette(1);

      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      TCanvas canvas("Canv", "Canv", 1400, 800);
      canvas.Divide(2, 1);
      canvas.cd();

      auto hBPix = std::make_shared<TH2F>("Correlation BPIX",
                                          Form("SiPixel Gain Calibration %s BPIx;per %s gains;per %s pedestals",
                                               (isForHLT_ ? "ForHLT" : "Offline"),
                                               (isForHLT_ ? "column" : "pixel"),
                                               (isForHLT_ ? "column" : "pixel")),
                                          200,
                                          payload->getGainLow(),
                                          payload->getGainHigh(),
                                          200,
                                          payload->getPedLow(),
                                          payload->getPedHigh());

      auto hFPix = std::make_shared<TH2F>("Correlation FPIX",
                                          Form("SiPixel Gain Calibration %s FPix;per %s gains;per %s pedestals",
                                               (isForHLT_ ? "ForHLT" : "Offline"),
                                               (isForHLT_ ? "column" : "pixel"),
                                               (isForHLT_ ? "column" : "pixel")),
                                          200,
                                          payload->getGainLow(),
                                          payload->getGainHigh(),
                                          200,
                                          payload->getPedLow(),
                                          payload->getPedHigh());

      for (unsigned int i : {1, 2}) {
        SiPixelPI::adjustCanvasMargins(canvas.cd(i), 0.04, 0.12, 0.15, 0.13);
        canvas.cd(i)->Modified();
      }

      // actually fill the histograms
      fillTheHistos(payload, hBPix, hFPix, gainCalibPI::t_correlation);

      canvas.cd(1)->SetLogz();
      hBPix->SetTitle("");
      hBPix->Draw("colz");

      SiPixelPI::makeNicePlotStyle(hBPix.get());
      hBPix->GetYaxis()->SetTitleOffset(1.65);

      canvas.cd(2)->SetLogz();
      hFPix->SetTitle("");
      hFPix->Draw("colz");

      SiPixelPI::makeNicePlotStyle(hFPix.get());
      hFPix->GetYaxis()->SetTitleOffset(1.65);
      canvas.Update();

      TLegend legend = TLegend(0.3, 0.92, 0.70, 0.95);
      legend.SetHeader(("Payload hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      legend.SetLineColor(10);
      legend.SetTextSize(0.025);
      canvas.cd(1);
      legend.Draw("same");
      canvas.cd(2);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.045);
      ltx.SetTextAlign(11);
      canvas.cd(1);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Gain Calibration IOV:" + std::to_string(std::get<0>(iov))).c_str());

      ltx.DrawLatexNDC(0.75, 0.15, "BPIX");

      canvas.cd(2);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Gain Calibration IOV:" + std::to_string(std::get<0>(iov))).c_str());

      ltx.DrawLatexNDC(0.75, 0.15, "FPIX");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("out.root");
#endif
      return true;
    }

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /*******************************************************************
    1d histogram of SiPixelGainCalibration for Pedestals of 1 IOV
  ********************************************************************/
  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationValuesByPart
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGainCalibrationValuesByPart()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              Form("SiPixelGainCalibrationOffline %s Values By Partition", TypeName[myType])) {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      gStyle->SetOptStat("emr");

      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      TCanvas canvas("Canv", "Canv", 1400, 800);
      canvas.Divide(2, 1);
      canvas.cd();

      float minimum(9999.);
      float maximum(-9999.);

      switch (myType) {
        case gainCalibPI::t_gain:
          maximum = payload->getGainHigh();
          minimum = payload->getGainLow();
          break;
        case gainCalibPI::t_pedestal:
          maximum = payload->getPedHigh();
          minimum = payload->getPedLow();
          break;
        default:
          edm::LogError(label_) << "Unrecognized type " << myType << std::endl;
          break;
      }

      auto hBPix = std::make_shared<TH1F>(Form("%s BPIX", TypeName[myType]),
                                          Form("SiPixel Gain Calibration %s BPIx -%s;per %s %s (BPix);# %ss",
                                               (isForHLT_ ? "ForHLT" : "Offline"),
                                               TypeName[myType],
                                               (isForHLT_ ? "Column" : "Pixel"),
                                               TypeName[myType],
                                               (isForHLT_ ? "column" : "pixel")),
                                          200,
                                          minimum,
                                          maximum);

      auto hFPix = std::make_shared<TH1F>(Form("%s FPIX", TypeName[myType]),
                                          Form("SiPixel Gain Calibration %s FPix -%s;per %s %s (FPix);# %ss",
                                               (isForHLT_ ? "ForHLT" : "Offline"),
                                               TypeName[myType],
                                               (isForHLT_ ? "Column" : "Pixel"),
                                               TypeName[myType],
                                               (isForHLT_ ? "column" : "pixel")),
                                          200,
                                          minimum,
                                          maximum);

      for (unsigned int i : {1, 2}) {
        SiPixelPI::adjustCanvasMargins(canvas.cd(i), 0.04, 0.12, 0.12, 0.02);
        canvas.cd(i)->Modified();
      }

      // actually fill the histograms
      fillTheHistos(payload, hBPix, hFPix, myType);

      canvas.cd(1)->SetLogy();
      hBPix->SetTitle("");
      hBPix->GetYaxis()->SetRangeUser(0.1, hBPix->GetMaximum() * 10);
      hBPix->SetFillColor(kBlue);
      hBPix->SetMarkerStyle(20);
      hBPix->SetMarkerSize(1);
      hBPix->Draw("hist");

      SiPixelPI::makeNicePlotStyle(hBPix.get());
      hBPix->SetStats(true);

      canvas.cd(2)->SetLogy();
      hFPix->SetTitle("");
      hFPix->GetYaxis()->SetRangeUser(0.1, hFPix->GetMaximum() * 10);
      hFPix->SetFillColor(kBlue);
      hFPix->SetMarkerStyle(20);
      hFPix->SetMarkerSize(1);
      hFPix->Draw("hist");

      SiPixelPI::makeNicePlotStyle(hFPix.get());
      hFPix->SetStats(true);

      canvas.Update();

      TLegend legend = TLegend(0.32, 0.92, 0.97, 0.95);
      legend.SetHeader(("Payload hash: #bf{" + (std::get<1>(iov)) + "}").c_str(),
                       "C");  // option "C" allows to center the header
      //legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetLineColor(10);
      legend.SetTextSize(0.025);
      canvas.cd(1);
      legend.Draw("same");
      canvas.cd(2);
      legend.Draw("same");

      canvas.cd(1);
      TPaveStats* st1 = (TPaveStats*)hBPix->FindObject("stats");
      st1->SetTextSize(0.03);
      SiPixelPI::adjustStats(st1, 0.13, 0.815, 0.44, 0.915);

      canvas.cd(2);
      TPaveStats* st2 = (TPaveStats*)hFPix->FindObject("stats");
      st2->SetTextSize(0.03);
      SiPixelPI::adjustStats(st2, 0.14, 0.815, 0.44, 0.915);

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.045);
      ltx.SetTextAlign(11);
      canvas.cd(1);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Gain Calibration IOV:" + std::to_string(std::get<0>(iov))).c_str());

      canvas.cd(2);
      ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Gain Calibration IOV:" + std::to_string(std::get<0>(iov))).c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /************************************************
    1d histogram comparison of SiPixelGainCalibration
  *************************************************/
  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationValueComparisonBase : public cond::payloadInspector::PlotImage<PayloadType> {
  public:
    SiPixelGainCalibrationValueComparisonBase()
        : cond::payloadInspector::PlotImage<PayloadType>(
              Form("SiPixelGainCalibration %s Values Comparison", TypeName[myType])) {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
      } else {
        isForHLT_ = true;
      }
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      gStyle->SetOptStat("emr");
      TGaxis::SetExponentOffset(-0.1, 0.01, "y");  // Y offset
      TH1F::SetDefaultSumw2(true);

      std::vector<std::tuple<cond::Time_t, cond::Hash>> sorted_iovs = iovs;
      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const& t1, auto const& t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });
      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<PayloadType> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<PayloadType> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      float minimum(9999.);
      float maximum(-9999.);

      switch (myType) {
        case gainCalibPI::t_gain:
          maximum = std::max(last_payload->getGainHigh(), first_payload->getGainHigh());
          minimum = std::min(last_payload->getGainLow(), first_payload->getGainLow());
          break;
        case gainCalibPI::t_pedestal:
          maximum = std::max(last_payload->getPedHigh(), first_payload->getPedHigh());
          minimum = std::min(last_payload->getPedLow(), first_payload->getPedLow());
          break;
        default:
          edm::LogError(label_) << "Unrecognized type " << myType << std::endl;
          break;
      }

      TCanvas canvas("Canv", "Canv", 1200, 1000);
      canvas.cd();
      auto hfirst = std::make_shared<TH1F>(Form("First, IOV %s", firstIOVsince.c_str()),
                                           Form("SiPixel Gain Calibration %s - %s;per %s %s;# %ss",
                                                (isForHLT_ ? "ForHLT" : "Offline"),
                                                TypeName[myType],
                                                (isForHLT_ ? "Column" : "Pixel"),
                                                TypeName[myType],
                                                (isForHLT_ ? "column" : "pixel")),
                                           200,
                                           minimum,
                                           maximum);

      auto hlast = std::make_shared<TH1F>(Form("Last, IOV %s", lastIOVsince.c_str()),
                                          Form("SiPixel Gain Calibration %s - %s;per %s %s;# %ss",
                                               (isForHLT_ ? "ForHLT" : "Offline"),
                                               TypeName[myType],
                                               (isForHLT_ ? "Column" : "Pixel"),
                                               TypeName[myType],
                                               (isForHLT_ ? "column" : "pixel")),
                                          200,
                                          minimum,
                                          maximum);

      SiPixelPI::adjustCanvasMargins(canvas.cd(), 0.05, 0.12, 0.12, 0.03);
      canvas.Modified();

      gainCalibPI::fillTheHisto(first_payload, hfirst, myType);
      gainCalibPI::fillTheHisto(last_payload, hlast, myType);

      canvas.cd()->SetLogy();
      auto extrema = SiPixelPI::getExtrema(hfirst.get(), hlast.get());
      //hfirst->GetYaxis()->SetRangeUser(extrema.first, extrema.second * 1.10);
      hfirst->GetYaxis()->SetRangeUser(1., extrema.second * 10);

      hfirst->SetTitle("");
      hfirst->SetLineColor(kRed);
      hfirst->SetBarWidth(0.95);
      hfirst->Draw("hist");

      hlast->SetTitle("");
      hlast->SetFillColorAlpha(kBlue, 0.20);
      hlast->SetBarWidth(0.95);
      hlast->Draw("histsames");

      SiPixelPI::makeNicePlotStyle(hfirst.get());
      hfirst->SetStats(true);
      SiPixelPI::makeNicePlotStyle(hlast.get());
      hlast->SetStats(true);

      canvas.Update();

      TLegend legend = TLegend(0.45, 0.86, 0.74, 0.94);
      //legend.SetHeader("#font[22]{SiPixel Offline Gain Calibration Comparison}", "C");  // option "C" allows to center the header
      //legend.AddEntry(hfirst.get(), ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "FL");
      //legend.AddEntry(hlast.get(),  ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "FL");
      legend.AddEntry(hfirst.get(), ("payload: #color[2]{" + std::get<1>(firstiov) + "}").c_str(), "F");
      legend.AddEntry(hlast.get(), ("payload: #color[4]{" + std::get<1>(lastiov) + "}").c_str(), "F");
      legend.SetTextSize(0.022);
      legend.SetLineColor(10);
      legend.Draw("same");

      TPaveStats* st1 = (TPaveStats*)hfirst->FindObject("stats");
      st1->SetTextSize(0.022);
      st1->SetLineColor(kRed);
      st1->SetTextColor(kRed);
      SiPixelPI::adjustStats(st1, 0.13, 0.84, 0.31, 0.94);

      TPaveStats* st2 = (TPaveStats*)hlast->FindObject("stats");
      st2->SetTextSize(0.022);
      st2->SetLineColor(kBlue);
      st2->SetTextColor(kBlue);
      SiPixelPI::adjustStats(st2, 0.13, 0.73, 0.31, 0.83);

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      //ltx.SetTextColor(kBlue);
      ltx.SetTextSize(0.047);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("SiPixel Gain Calibration IOV:#color[2]{" + std::to_string(std::get<0>(firstiov)) +
                        "} vs IOV:#color[4]{" + std::to_string(std::get<0>(lastiov)) + "}")
                           .c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("out.root");
#endif

      return true;
    }

  protected:
    bool isForHLT_;
    std::string label_;
  };

  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationValueComparisonSingleTag
      : public SiPixelGainCalibrationValueComparisonBase<myType, PayloadType> {
  public:
    SiPixelGainCalibrationValueComparisonSingleTag()
        : SiPixelGainCalibrationValueComparisonBase<myType, PayloadType>() {
      this->setSingleIov(false);
    }
  };

  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationValueComparisonTwoTags
      : public SiPixelGainCalibrationValueComparisonBase<myType, PayloadType> {
  public:
    SiPixelGainCalibrationValueComparisonTwoTags() : SiPixelGainCalibrationValueComparisonBase<myType, PayloadType>() {
      this->setTwoTags(true);
    }
  };

  // 2D MAPS

  /************************************************
   occupancy style map BPix
  *************************************************/
  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationBPIXMap
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGainCalibrationBPIXMap()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              Form("SiPixelGainCalibration %s Barrel Pixel Map", TypeName[myType])),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    };

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      static const int n_layers = 4;
      int nlad_list[n_layers] = {6, 14, 22, 32};
      int divide_roc = 1;

      // ---------------------    BOOK HISTOGRAMS
      std::array<TH2D*, n_layers> h_bpix_Gains;

      for (unsigned int lay = 1; lay <= 4; lay++) {
        int nlad = nlad_list[lay - 1];

        std::string name = "occ_Gain_Layer_" + std::to_string(lay);
        std::string title = "; Module # ; Ladder #";
        h_bpix_Gains[lay - 1] = new TH2D(name.c_str(),
                                         title.c_str(),
                                         72 * divide_roc,
                                         -4.5,
                                         4.5,
                                         (nlad * 4 + 2) * divide_roc,
                                         -nlad - 0.5,
                                         nlad + 0.5);
      }

      std::map<uint32_t, float> GainCalibMap_;
      gainCalibPI::fillThePerModuleMap(payload, GainCalibMap_, myType);
      if (GainCalibMap_.size() != SiPixelPI::phase1size) {
        edm::LogError(label_) << "SiPixelGainCalibration maps are not supported for non-Phase1 Pixel geometries !";
        TCanvas canvas("Canv", "Canv", 1200, 1000);
        SiPixelPI::displayNotSupported(canvas, GainCalibMap_.size());
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      // hard-coded phase-I
      std::array<double, 4> minima = {{999., 999., 999., 999.}};

      for (const auto& element : GainCalibMap_) {
        int subid = DetId(element.first).subdetId();
        if (subid == PixelSubdetector::PixelBarrel) {
          auto layer = m_trackerTopo.pxbLayer(DetId(element.first));
          auto s_ladder = SiPixelPI::signed_ladder(DetId(element.first), m_trackerTopo, true);
          auto s_module = SiPixelPI::signed_module(DetId(element.first), m_trackerTopo, true);

          auto ladder = m_trackerTopo.pxbLadder(DetId(element.first));
          auto module = m_trackerTopo.pxbModule(DetId(element.first));
          COUT << "layer:" << layer << " ladder:" << ladder << " module:" << module << " signed ladder: " << s_ladder
               << " signed module: " << s_module << std::endl;

          if (element.second < minima.at(layer - 1))
            minima.at(layer - 1) = element.second;

          auto rocsToMask = SiPixelPI::maskedBarrelRocsToBins(layer, s_ladder, s_module);
          for (const auto& bin : rocsToMask) {
            h_bpix_Gains[layer - 1]->SetBinContent(bin.first, bin.second, element.second);
          }
        }
      }

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 1200);
      canvas.Divide(2, 2);

      for (unsigned int lay = 1; lay <= 4; lay++) {
        SiPixelPI::adjustCanvasMargins(canvas.cd(lay), -1, 0.08, 0.1, 0.13);

        COUT << " layer:" << lay << " max:" << h_bpix_Gains[lay - 1]->GetMaximum() << " min: " << minima.at(lay - 1)
             << std::endl;

        SiPixelPI::dress_occup_plot(canvas, h_bpix_Gains[lay - 1], lay, 0, 1, true, true, false);
        h_bpix_Gains[lay - 1]->GetZaxis()->SetRangeUser(minima.at(lay - 1) - 0.001,
                                                        h_bpix_Gains[lay - 1]->GetMaximum() + 0.001);
      }

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      for (unsigned int lay = 1; lay <= 4; lay++) {
        canvas.cd(lay);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.055);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /************************************************
   occupancy style map FPix
  *************************************************/

  template <gainCalibPI::type myType, class PayloadType>
  class SiPixelGainCalibrationFPIXMap
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGainCalibrationFPIXMap()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              Form("SiPixelGainCalibration %s Forward Pixel Map", TypeName[myType])),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      static const int n_rings = 2;
      std::array<TH2D*, n_rings> h_fpix_Gains;
      int divide_roc = 1;

      // ---------------------    BOOK HISTOGRAMS
      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        int n = ring == 1 ? 92 : 140;
        float y = ring == 1 ? 11.5 : 17.5;
        std::string name = "occ_Gains_ring_" + std::to_string(ring);
        std::string title = "; Disk # ; Blade/Panel #";

        h_fpix_Gains[ring - 1] =
            new TH2D(name.c_str(), title.c_str(), 56 * divide_roc, -3.5, 3.5, n * divide_roc, -y, y);
      }

      std::map<uint32_t, float> GainCalibMap_;
      gainCalibPI::fillThePerModuleMap(payload, GainCalibMap_, myType);
      if (GainCalibMap_.size() != SiPixelPI::phase1size) {
        edm::LogError(label_) << "SiPixelGainCalibration maps are not supported for non-Phase1 Pixel geometries !";
        TCanvas canvas("Canv", "Canv", 1200, 1000);
        SiPixelPI::displayNotSupported(canvas, GainCalibMap_.size());
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
        return false;
      }

      // hardcoded phase-I
      std::array<double, 2> minima = {{999., 999.}};

      for (const auto& element : GainCalibMap_) {
        int subid = DetId(element.first).subdetId();
        if (subid == PixelSubdetector::PixelEndcap) {
          auto ring = SiPixelPI::ring(DetId(element.first), m_trackerTopo, true);
          auto s_blade = SiPixelPI::signed_blade(DetId(element.first), m_trackerTopo, true);
          auto s_disk = SiPixelPI::signed_disk(DetId(element.first), m_trackerTopo, true);
          auto s_blade_panel = SiPixelPI::signed_blade_panel(DetId(element.first), m_trackerTopo, true);
          auto panel = m_trackerTopo.pxfPanel(element.first);

          COUT << "ring:" << ring << " blade: " << s_blade << " panel: " << panel
               << " signed blade/panel: " << s_blade_panel << " disk: " << s_disk << std::endl;

          if (element.second < minima.at(ring - 1))
            minima.at(ring - 1) = element.second;

          auto rocsToMask = SiPixelPI::maskedForwardRocsToBins(ring, s_blade, panel, s_disk);
          for (const auto& bin : rocsToMask) {
            h_fpix_Gains[ring - 1]->SetBinContent(bin.first, bin.second, element.second);
          }
        }
      }

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 600);
      canvas.Divide(2, 1);

      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        SiPixelPI::adjustCanvasMargins(canvas.cd(ring), -1, 0.08, 0.1, 0.13);

        COUT << " ringer:" << ring << " max:" << h_fpix_Gains[ring - 1]->GetMaximum() << " min: " << minima.at(ring - 1)
             << std::endl;

        SiPixelPI::dress_occup_plot(canvas, h_fpix_Gains[ring - 1], 0, ring, 1, true, true, false);
        h_fpix_Gains[ring - 1]->GetZaxis()->SetRangeUser(minima.at(ring - 1) - 0.001,
                                                         h_fpix_Gains[ring - 1]->GetMaximum() + 0.001);
      }

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      for (unsigned int ring = 1; ring <= n_rings; ring++) {
        canvas.cd(ring);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.05);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      return true;
    }

  private:
    TrackerTopology m_trackerTopo;

  protected:
    bool isForHLT_;
    std::string label_;
  };

  /************************************************
   Summary Comparison per region of SiPixelGainCalibration between 2 IOVs
  *************************************************/
  template <gainCalibPI::type myType, class PayloadType, cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class SiPixelGainCalibrationByRegionComparisonBase
      : public cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags> {
  public:
    SiPixelGainCalibrationByRegionComparisonBase()
        : cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags>(
              Form("SiPixelGainCalibration %s Comparison by Region", TypeName[myType])) {
      if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationOffline>) {
        isForHLT_ = false;
        label_ = "SiPixelGainCalibrationOffline_PayloadInspector";
      } else {
        isForHLT_ = true;
        label_ = "SiPixelGainCalibrationForHLT_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetPaintTextFormat(".3f");

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<PayloadType> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<PayloadType> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::map<uint32_t, float> f_GainsMap_;
      gainCalibPI::fillThePerModuleMap(first_payload, f_GainsMap_, myType);

      std::map<uint32_t, float> l_GainsMap_;
      gainCalibPI::fillThePerModuleMap(last_payload, l_GainsMap_, myType);

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Comparison", "Comparison", 1600, 800);

      SiPixelPI::PhaseInfo f_phaseInfo(f_GainsMap_.size());
      SiPixelPI::PhaseInfo l_phaseInfo(l_GainsMap_.size());

      std::map<SiPixelPI::regions, std::shared_ptr<TH1F>> FirstGains_spectraByRegion;
      std::map<SiPixelPI::regions, std::shared_ptr<TH1F>> LastGains_spectraByRegion;
      std::shared_ptr<TH1F> summaryFirst;
      std::shared_ptr<TH1F> summaryLast;

      float minimum(9999.);
      float maximum(-9999.);

      switch (myType) {
        case gainCalibPI::t_gain:
          maximum = std::max(last_payload->getGainHigh(), first_payload->getGainHigh());
          minimum = std::min(last_payload->getGainLow(), first_payload->getGainLow());
          break;
        case gainCalibPI::t_pedestal:
          maximum = std::max(last_payload->getPedHigh(), first_payload->getPedHigh());
          minimum = std::min(last_payload->getPedLow(), first_payload->getPedLow());
          break;
        default:
          edm::LogError(label_) << "Unrecognized type " << myType << std::endl;
          break;
      }

      // book the intermediate histograms
      for (int r = SiPixelPI::BPixL1o; r != SiPixelPI::NUM_OF_REGIONS; r++) {
        SiPixelPI::regions part = static_cast<SiPixelPI::regions>(r);
        std::string s_part = SiPixelPI::getStringFromRegionEnum(part);

        FirstGains_spectraByRegion[part] =
            std::make_shared<TH1F>(Form("hfirstGains_%s", s_part.c_str()),
                                   Form(";%s #LT %s #GT;n. of modules", s_part.c_str(), TypeName[myType]),
                                   200,
                                   minimum,
                                   maximum);

        LastGains_spectraByRegion[part] =
            std::make_shared<TH1F>(Form("hlastGains_%s", s_part.c_str()),
                                   Form(";%s #LT %s #GT;n. of modules", s_part.c_str(), TypeName[myType]),
                                   200,
                                   minimum,
                                   maximum);
      }

      summaryFirst = std::make_shared<TH1F>("first Summary",
                                            Form("Summary of #LT per %s %s #GT;;average %s",
                                                 (isForHLT_ ? "Column" : "Pixel"),
                                                 TypeName[myType],
                                                 TypeName[myType]),
                                            FirstGains_spectraByRegion.size(),
                                            0,
                                            FirstGains_spectraByRegion.size());

      summaryLast = std::make_shared<TH1F>("last Summary",
                                           Form("Summary of #LT per %s %s #GT;;average %s",
                                                (isForHLT_ ? "Column" : "Pixel"),
                                                TypeName[myType],
                                                TypeName[myType]),
                                           LastGains_spectraByRegion.size(),
                                           0,
                                           LastGains_spectraByRegion.size());

      // deal with first IOV
      const char* path_toTopologyXML = f_phaseInfo.pathToTopoXML();

      auto f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      // -------------------------------------------------------------------
      // loop on the first Gains Map
      // -------------------------------------------------------------------
      for (const auto& it : f_GainsMap_) {
        if (DetId(it.first).det() != DetId::Tracker) {
          edm::LogWarning(label_) << "Encountered invalid Tracker DetId:" << it.first << " - terminating ";
          return false;
        }

        SiPixelPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.first);
        t_info_fromXML.fillGeometryInfo(detid, f_tTopo, f_phaseInfo.phase());

        SiPixelPI::regions thePart = t_info_fromXML.filterThePartition();
        FirstGains_spectraByRegion[thePart]->Fill(it.second);
      }  // ends loop on the vector of error transforms

      // deal with last IOV
      path_toTopologyXML = l_phaseInfo.pathToTopoXML();

      auto l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      // -------------------------------------------------------------------
      // loop on the second Gains Map
      // -------------------------------------------------------------------
      for (const auto& it : l_GainsMap_) {
        if (DetId(it.first).det() != DetId::Tracker) {
          edm::LogWarning(label_) << "Encountered invalid Tracker DetId:" << it.first << " - terminating ";
          return false;
        }

        SiPixelPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.first);
        t_info_fromXML.fillGeometryInfo(detid, l_tTopo, l_phaseInfo.phase());

        SiPixelPI::regions thePart = t_info_fromXML.filterThePartition();
        LastGains_spectraByRegion[thePart]->Fill(it.second);
      }  // ends loop on the vector of error transforms

      // fill the summary plots
      int bin = 1;
      for (int r = SiPixelPI::BPixL1o; r != SiPixelPI::NUM_OF_REGIONS; r++) {
        SiPixelPI::regions part = static_cast<SiPixelPI::regions>(r);

        summaryFirst->GetXaxis()->SetBinLabel(bin, SiPixelPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float f_mean =
            FirstGains_spectraByRegion[part]->GetMean() > 10.e-6 ? FirstGains_spectraByRegion[part]->GetMean() : 0.;
        summaryFirst->SetBinContent(bin, f_mean);
        //summaryFirst->SetBinError(bin,Gains_spectraByRegion[hash]->GetRMS());

        summaryLast->GetXaxis()->SetBinLabel(bin, SiPixelPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float l_mean =
            LastGains_spectraByRegion[part]->GetMean() > 10.e-6 ? LastGains_spectraByRegion[part]->GetMean() : 0.;
        summaryLast->SetBinContent(bin, l_mean);
        //summaryLast->SetBinError(bin,Gains_spectraByRegion[hash]->GetRMS());
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

      SiPixelPI::adjustCanvasMargins(canvas.cd(), -1, 0.18, 0.11, 0.02);
      canvas.Modified();

      summaryFirst->SetFillColor(kRed);
      summaryLast->SetFillColor(kBlue);

      summaryFirst->SetBarWidth(0.45);
      summaryFirst->SetBarOffset(0.1);

      summaryLast->SetBarWidth(0.4);
      summaryLast->SetBarOffset(0.55);

      summaryLast->SetMarkerSize(1.2);
      summaryFirst->SetMarkerSize(1.2);

      float max = (summaryFirst->GetMaximum() > summaryLast->GetMaximum()) ? summaryFirst->GetMaximum()
                                                                           : summaryLast->GetMaximum();

      summaryFirst->GetYaxis()->SetRangeUser(0., std::max(0., max * 1.40));

      summaryFirst->Draw("b text0");
      summaryLast->Draw("b text0 same");

      TLegend legend = TLegend(0.52, 0.80, 0.98, 0.9);
      legend.SetHeader(Form("#LT %s #GT value comparison", TypeName[myType]),
                       "C");  // option "C" allows to center the header

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

  protected:
    bool isForHLT_;
    std::string label_;
  };
}  // namespace gainCalibHelper

#endif
