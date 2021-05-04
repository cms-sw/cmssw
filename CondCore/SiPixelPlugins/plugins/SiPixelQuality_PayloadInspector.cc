/*!
  \file SiPixelQuality_PayloadInspector
  \Payload Inspector Plugin for SiPixelQuality
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/10/18 14:48:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  /************************************************
    test class
  *************************************************/

  class SiPixelQualityTest
      : public cond::payloadInspector::Histogram1D<SiPixelQuality, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelQualityTest()
        : cond::payloadInspector::Histogram1D<SiPixelQuality, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelQuality test", "SiPixelQuality test", 10, 0.0, 10.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiPixelQuality> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          auto theDisabledModules = payload->getBadComponentList();
          for (const auto& mod : theDisabledModules) {
            int BadRocCount(0);
            for (unsigned short n = 0; n < 16; n++) {
              unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
              if (mod.BadRocs & mask)
                BadRocCount++;
            }
            COUT << "detId:" << mod.DetID << " error type:" << mod.errorType << " BadRocs:" << BadRocCount << std::endl;
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
    summary class
  *************************************************/

  class SiPixelQualityBadRocsSummary
      : public cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::MULTI_IOV, 1> {
  public:
    SiPixelQualityBadRocsSummary()
        : cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::MULTI_IOV, 1>(
              "SiPixel Quality Summary") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (const auto& iov : tag.iovs) {
        std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));
        auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

        COUT << "======================= " << unpacked.first << " : " << unpacked.second << std::endl;
        auto theDisabledModules = payload->getBadComponentList();
        for (const auto& mod : theDisabledModules) {
          COUT << "detId: " << mod.DetID << " |error type: " << mod.errorType << " |BadRocs: " << mod.BadRocs
               << std::endl;
        }
      }

      //=========================
      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      SiPixelPI::displayNotSupported(canvas, 0);
      canvas.cd();
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
    time history class
  *************************************************/

  class SiPixelQualityBadRocsTimeHistory
      : public cond::payloadInspector::TimeHistoryPlot<SiPixelQuality, std::pair<double, double> > {
  public:
    SiPixelQualityBadRocsTimeHistory()
        : cond::payloadInspector::TimeHistoryPlot<SiPixelQuality, std::pair<double, double> >("bad ROCs count vs time",
                                                                                              "bad ROCs count") {}

    std::pair<double, double> getFromPayload(SiPixelQuality& payload) override {
      return std::make_pair(extractBadRocCount(payload), 0.);
    }

    unsigned int extractBadRocCount(SiPixelQuality& payload) {
      unsigned int BadRocCount(0);
      auto theDisabledModules = payload.getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        for (unsigned short n = 0; n < 16; n++) {
          unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
          if (mod.BadRocs & mask)
            BadRocCount++;
        }
      }
      return BadRocCount;
    }
  };

  /************************************************
   occupancy style map BPix
  *************************************************/

  class SiPixelBPixQualityMap
      : public cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelBPixQualityMap()
        : cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelQuality Barrel Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));

      Phase1PixelROCMaps theBPixMap("");

      auto theDisabledModules = payload->getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        int subid = DetId(mod.DetID).subdetId();
        std::bitset<16> bad_rocs(mod.BadRocs);
        if (subid == PixelSubdetector::PixelBarrel) {
          if (payload->IsModuleBad(mod.DetID)) {
            theBPixMap.fillWholeModule(mod.DetID, 1.);
          } else {
            theBPixMap.fillSelectedRocs(mod.DetID, bad_rocs, 1.);
          }
        }
      }

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 1200);
      theBPixMap.drawBarrelMaps(canvas);

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

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outBPix.root");
#endif

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
   occupancy style map FPix
  *************************************************/

  class SiPixelFPixQualityMap
      : public cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelFPixQualityMap()
        : cond::payloadInspector::PlotImage<SiPixelQuality, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelQuality Forward Pixel Map"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelQuality> payload = fetchPayload(std::get<1>(iov));

      Phase1PixelROCMaps theFPixMap("");

      auto theDisabledModules = payload->getBadComponentList();
      for (const auto& mod : theDisabledModules) {
        int subid = DetId(mod.DetID).subdetId();
        std::bitset<16> bad_rocs(mod.BadRocs);
        if (subid == PixelSubdetector::PixelEndcap) {
          if (payload->IsModuleBad(mod.DetID)) {
            theFPixMap.fillWholeModule(mod.DetID, 1.);
          } else {
            theFPixMap.fillSelectedRocs(mod.DetID, bad_rocs, 1.);
          }
        }  // if it's endcap
      }    // loop on disable moduels

      gStyle->SetOptStat(0);
      //=========================
      TCanvas canvas("Summary", "Summary", 1200, 600);
      theFPixMap.drawForwardMaps(canvas);

      auto unpacked = SiPixelPI::unpack(std::get<0>(iov));

      for (unsigned int ring = 1; ring <= 2; ring++) {
        canvas.cd(ring);
        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.050);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         unpacked.first == 0
                             ? ("IOV:" + std::to_string(unpacked.second)).c_str()
                             : (std::to_string(unpacked.first) + "," + std::to_string(unpacked.second)).c_str());
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
#ifdef MMDEBUG
      canvas.SaveAs("outFPix.root");
#endif
      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQuality) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsSummary);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsTimeHistory);
  PAYLOAD_INSPECTOR_CLASS(SiPixelBPixQualityMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelFPixQualityMap);
}
