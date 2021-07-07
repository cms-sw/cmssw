#include "catch.hpp"
#include <iostream>
#include <numeric>  // std::accumulate
#include "TCanvas.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"

static const std::string k_geo = "SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt";

TEST_CASE("Phase1PixelROCMaps testing", "[Phase1PixelROCMaps]") {
  //_____________________________________________________________
  SECTION("Check barrel plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelROCMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.drawBarrelMaps(c, "testing barrel");
    c.SaveAs("Phase1PixelROCMaps_barrel.png");
    REQUIRE(theMap.getLayerMaps().size() == 4);
  }

  //_____________________________________________________________
  SECTION("Check endcap plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelROCMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 600);
    theMap.drawForwardMaps(c, "testing endcaps");
    c.SaveAs("Phase1PixelROCMaps_endcap.png");
    REQUIRE(theMap.getRingMaps().size() == 2);
  }

  //_____________________________________________________________
  SECTION("Check whole detector plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelROCMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1600);
    theMap.drawMaps(c, "testing everything");
    c.SaveAs("Phase1PixelROCMaps_whole.png");
    REQUIRE(theMap.getLayerMaps().size() == 4);
    REQUIRE(theMap.getRingMaps().size() == 2);
  }

  //_____________________________________________________________
  SECTION("Check empty delta plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelROCMaps theMap("#Delta", "#Delta");
    TCanvas c = TCanvas("c", "c", 1200, 1600);
    theMap.drawMaps(c, "testing empty #Delta");
    theMap.fillWholeModule(303042564, 1.);
    theMap.fillWholeModule(344912900, -1.);
    c.SaveAs("Phase1PixelROCMaps_emptyDelta.png");
    REQUIRE(theMap.getLayerMaps().size() == 4);
    REQUIRE(theMap.getRingMaps().size() == 2);
  }

  //_____________________________________________________________
  SECTION("Check filling whole modules") {
    Phase1PixelROCMaps theMap("");
    gStyle->SetOptStat(0);
    TCanvas c = TCanvas("c", "c", 1200, 1600);
    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const auto& detIds = reader_.getAllDetIds();
    for (const auto& it : detIds) {
      int subid = DetId(it).subdetId();
      if (subid == PixelSubdetector::PixelBarrel) {
        int module = theMap.findDetCoordinates(it).m_s_module;
        if (module % 2 == 0) {
          theMap.fillWholeModule(it, 1.);
        } else {
          theMap.fillWholeModule(it, -1.);
        }
      } else if (subid == PixelSubdetector::PixelEndcap) {
        int panel = theMap.findDetCoordinates(it).m_panel;
        if (panel % 2 == 0) {
          theMap.fillWholeModule(it, 1.);
        } else {
          theMap.fillWholeModule(it, -1.);
        }
      }
    }
    theMap.drawMaps(c, "testing whole modules");
    c.SaveAs("Phase1PixelROCMaps_fullModules.png");

    int totalEntries = 0;
    const auto layerMaps = theMap.getLayerMaps();
    const auto ringMaps = theMap.getRingMaps();

    int nBarrel = std::accumulate(layerMaps.begin(), layerMaps.end(), 0, [](int a, const std::shared_ptr<TH2> h) {
      return a += h.get()->GetEntries();
    });

    int nForward = std::accumulate(ringMaps.begin(), ringMaps.end(), 0, [](int a, const std::shared_ptr<TH2> h) {
      return a += h.get()->GetEntries();
    });

    totalEntries = nBarrel + nForward;
    // this must equal the total number of modules (1856) time the ROC/module (16)
    REQUIRE(totalEntries == (1856 * 16));
  }

  //_____________________________________________________________
  SECTION("Check filling in delta mode") {
    Phase1PixelROCMaps theMap("", "#Delta: flipped vs unflipped");
    gStyle->SetOptStat(0);
    TCanvas c = TCanvas("c", "c", 1200, 1600);
    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const auto& detIds = reader_.getAllDetIds();
    for (const auto& it : detIds) {
      bool isFlipped = theMap.findDetCoordinates(it).isFlipped();
      theMap.fillWholeModule(it, isFlipped ? 1. : -1);
    }
    theMap.drawMaps(c, "testing #Delta mode");
    c.SaveAs("Phase1PixelROCMaps_deltaMode.png");

    int totalEntries = 0;
    const auto layerMaps = theMap.getLayerMaps();
    const auto ringMaps = theMap.getRingMaps();

    int nBarrel = std::accumulate(layerMaps.begin(), layerMaps.end(), 0, [](int a, const std::shared_ptr<TH2> h) {
      return a += h.get()->GetEntries();
    });

    int nForward = std::accumulate(ringMaps.begin(), ringMaps.end(), 0, [](int a, const std::shared_ptr<TH2> h) {
      return a += h.get()->GetEntries();
    });

    totalEntries = nBarrel + nForward;
    // this must equal the total number of modules (1856) time the ROC/module (16)
    REQUIRE(totalEntries == (1856 * 16));
  }

  //_____________________________________________________________
  SECTION("Check filling selected ROCs") {
    Phase1PixelROCMaps theMap("");
    gStyle->SetOptStat(0);
    TCanvas c = TCanvas("c", "c", 1200, 1600);
    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const auto& detIds = reader_.getAllDetIds();
    for (const auto& it : detIds) {
      for (unsigned i = 0; i < 16; i++) {
        std::bitset<16> bad_rocs;
        bad_rocs.set(i);
        theMap.fillSelectedRocs(it, bad_rocs, i);
      }
      //std::bitset<16> bad_rocs(it >> 2);
      //theMap.fillSelectedRocs(it, bad_rocs, 1.);
    }

    int totalEntries = 0;
    const auto layerMaps = theMap.getLayerMaps();
    const auto ringMaps = theMap.getRingMaps();

    int nBarrel = std::accumulate(layerMaps.begin(), layerMaps.end(), 0, [](int a, const std::shared_ptr<TH2> h) {
      return a += h.get()->GetEntries();
    });

    int nForward = std::accumulate(ringMaps.begin(), ringMaps.end(), 0, [](int a, const std::shared_ptr<TH2> h) {
      return a += h.get()->GetEntries();
    });

    totalEntries = nBarrel + nForward;

    theMap.drawMaps(c, "testing selected ROCs");
    c.SaveAs("Phase1PixelROCMaps_fullROCs.png");
    // this must equal the total number of modules (1856) time the ROC/module (16)
    REQUIRE(totalEntries == (1856 * 16));
  }
}
