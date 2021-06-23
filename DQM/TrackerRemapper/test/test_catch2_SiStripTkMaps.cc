#include "catch.hpp"
#include <iostream>
#include <numeric>  // std::accumulate
#include "TCanvas.h"
#include "DQM/TrackerRemapper/interface/SiStripTkMaps.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

static const std::string k_geo = "CalibTracker/SiStripCommon/data/SiStripDetInfo.dat";

TEST_CASE("SiStripTkMaps testing", "[SiStripTkMaps]") {
  //_____________________________________________________________
  SECTION("Check SiStrip Tk Maps plotting") {
    gStyle->SetOptStat(0);
    SiStripTkMaps theMap("COLZA L");
    theMap.bookMap("testing SiStripTkMaps", "counts");
    SiStripDetInfoFileReader reader_ = SiStripDetInfoFileReader(edm::FileInPath(k_geo).fullPath());
    const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>& DetInfos = reader_.getAllData();
    unsigned int count = 0;
    for (const auto& it : DetInfos) {
      count++;
      theMap.fill(it.first, count);
    }

    const auto filledIds = theMap.getTheFilledIds();
    TCanvas c = TCanvas("c", "c");
    theMap.drawMap(c, "");
    c.SaveAs("SiStripsTkMaps.png");
    std::cout << "SiStripTkMaps filled " << filledIds.size() << " DetIds" << std::endl;
    REQUIRE(filledIds.size() == count);
  }
}
