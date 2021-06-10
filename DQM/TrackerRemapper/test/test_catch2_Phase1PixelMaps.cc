#include "catch.hpp"
#include <iostream>
#include <numeric>      // std::accumulate
#include "TCanvas.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"

TEST_CASE("Phase1PixelMaps testing", "[Phase1PixelMaps]") {
  //_____________________________________________________________
  SECTION("Check barrel plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookBarrelHistograms("mytest","test","test");
    theMap.bookBarrelBins("mytest");
    theMap.drawBarrelMaps("mytest",c,"colz0");
    c.SaveAs("Phase1PixelMaps_barrel.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check endcap plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookForwardHistograms("mytest","test","test");
    theMap.bookForwardBins("mytest");
    theMap.drawForwardMaps("mytest",c,"colz0");
    c.SaveAs("Phase1PixelMaps_endcap.png");
    REQUIRE(true);
  }

  //_____________________________________________________________
  SECTION("Check summary plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelMaps theMap("");
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.bookBarrelHistograms("mytest","test","test");
    theMap.bookBarrelBins("mytest");
    theMap.bookForwardHistograms("mytest","test","test");
    theMap.bookForwardBins("mytest");
    theMap.drawSummaryMaps("mytest",c);
    c.SaveAs("Phase1PixelMaps_summary.png");
    REQUIRE(true);
  }


}
