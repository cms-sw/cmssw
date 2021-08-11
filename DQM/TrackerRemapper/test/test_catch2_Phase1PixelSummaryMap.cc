#include "catch.hpp"
#include <iostream>
#include <numeric>  // std::accumulate
#include "TCanvas.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"

static const std::string k_geo = "SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt";

TEST_CASE("Phase1PixelSummaryMap testing", "[Phase1PixelSummaryMap]") {
  //_____________________________________________________________
  SECTION("Check Phase1Pixel Summary plotting") {
    gStyle->SetOptStat(0);
    Phase1PixelSummaryMap theMap("colz", "test", "testing");
    theMap.createTrackerBaseMap();
    SiPixelDetInfoFileReader reader_ = SiPixelDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const auto& detIds = reader_.getAllDetIds();
    int count = 0;
    for (const auto& it : detIds) {
      count++;
      theMap.fillTrackerMap(it, count);
    }
    TCanvas c = TCanvas("c", "c", 1200, 1200);
    theMap.printTrackerMap(c);
    c.SaveAs("Phase1PixelSummaryMap.png");
    REQUIRE(true);
  }
}
