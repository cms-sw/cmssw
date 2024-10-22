#include "catch.hpp"
#include <iostream>
#include <numeric>  // std::accumulate
#include "TCanvas.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"

static const std::string k_geo = "SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt";

TEST_CASE("Phase1PixelMaps testing", "[Phase1PixelMaps]") {
  //_____________________________________________________________
  SECTION("Check barrel plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookBarrelHistograms("mytest", "test", "z-axis");
    theMap.drawBarrelMaps("mytest", c, "colz0");
    theMap.beautifyAllHistograms();
    c.SaveAs("Phase1PixelMaps_Barrel.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check endcap plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 800);
    theMap.bookForwardHistograms("mytest", "test", "z-axis");
    theMap.drawForwardMaps("mytest", c, "colz0");
    theMap.beautifyAllHistograms();
    c.SaveAs("Phase1PixelMaps_Endcap.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("textAL");  // needed to not show the axis
    TCanvas c = TCanvas("c", "c", 1200, 800);
    theMap.bookBarrelHistograms("mytest", "test", "z-axis");
    theMap.bookForwardHistograms("mytest", "test", "z-axis");
    theMap.beautifyAllHistograms();
    theMap.drawSummaryMaps("mytest", c);
    c.SaveAs("Phase1PixelMaps_Summary.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary filling") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("COLZA L");
    TCanvas c = TCanvas("c", "c", 1200, 800);
    theMap.bookBarrelHistograms("mytest", "test", "z-axis");
    theMap.bookForwardHistograms("mytest", "test", "z-axis");

    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
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
    theMap.beautifyAllHistograms();
    theMap.drawSummaryMaps("mytest", c);
    c.SaveAs("Phase1PixelMaps_Summary_Filled.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary filling V2") {
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    Phase1PixelMaps theMap("COLZA L");

    TCanvas c = TCanvas("c", "c", 1200, 800);
    theMap.book("mytest", "module counts", "module counts");

    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const auto& detIds = reader_.getAllDetIds();
    int count = 0;
    for (const auto& it : detIds) {
      count++;
      theMap.fill("mytest", it, count);
    }

    theMap.beautifyAllHistograms();
    theMap.drawSummaryMaps("mytest", c);
    c.SaveAs("Phase1PixelMaps_Summary_Filled_V2.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary filling V3") {
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kBlackBody);
    Phase1PixelMaps theMap("COLZA L");

    TCanvas c = TCanvas("c", "c", 1200, 800);
    theMap.book("mytest", "module counts", "module counts");

    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const auto& detIds = reader_.getAllDetIds();
    int count = 0;
    for (const auto& it : detIds) {
      count++;
      theMap.fill("mytest", it, count);
    }

    theMap.setNoRescale();
    theMap.beautifyAllHistograms();
    theMap.drawSummaryMaps("mytest", c);
    theMap.setBarrelScale("mytest", std::make_pair(0., 100.));
    theMap.setForwardScale("mytest", std::make_pair(0., 20.));
    c.SaveAs("Phase1PixelMaps_Summary_Filled_V3.png");
    REQUIRE(true);
  }
}
