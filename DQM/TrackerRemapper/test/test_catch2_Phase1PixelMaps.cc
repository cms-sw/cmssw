#include "catch.hpp"
#include <iostream>
#include <numeric>  // std::accumulate
#include "TCanvas.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"

TEST_CASE("Phase1PixelMaps testing", "[Phase1PixelMaps]") {
  //_____________________________________________________________
  SECTION("Check barrel plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookBarrelHistograms("mytest", "test", "test");
    theMap.bookBarrelBins("mytest");
    theMap.drawBarrelMaps("mytest", c, "colz0");
    c.SaveAs("Phase1PixelMaps_barrel.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check endcap plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookForwardHistograms("mytest", "test", "test");
    theMap.bookForwardBins("mytest");
    theMap.drawForwardMaps("mytest", c, "colz0");
    c.SaveAs("Phase1PixelMaps_endcap.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookBarrelHistograms("mytest", "test", "test");
    theMap.bookBarrelBins("mytest");
    theMap.bookForwardHistograms("mytest", "test", "test");
    theMap.bookForwardBins("mytest");
    theMap.drawSummaryMaps("mytest", c);
    c.SaveAs("Phase1PixelMaps_summary.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary filling") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("COLZA L");
    TCanvas c = TCanvas("c", "c", 1200, 800);
    theMap.bookBarrelHistograms("mytest", "test", "test");
    theMap.bookBarrelBins("mytest");
    theMap.bookForwardHistograms("mytest", "test", "test");
    theMap.bookForwardBins("mytest");

    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(
        edm::FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt").fullPath());
    const auto& detIds = reader_.getAllDetIds();
    int count = 0;
    for (const auto& it : detIds) {
      count++;
      int subid = DetId(it).subdetId();
      if (subid == PixelSubdetector::PixelBarrel) {
        theMap.fillBarrelBin("mytest", it, count);
      } else if (subid == PixelSubdetector::PixelEndcap) {
        theMap.fillForwardBin("mytest", it, count);
      }
    }

    theMap.drawSummaryMaps("mytest", c);
    c.SaveAs("Phase1PixelMaps_summary_full.png");
    REQUIRE(true);
  }
}
