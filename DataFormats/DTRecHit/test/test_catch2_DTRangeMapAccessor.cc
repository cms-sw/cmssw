#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"

namespace {

  template <typename F>
  void loop_all_superlayers(F&& iF) {
    for (int wheel = DTChamberId::minWheelId; wheel <= DTChamberId::maxWheelId; ++wheel) {
      for (int station = DTChamberId::minStationId; station <= DTChamberId::maxStationId; ++station) {
        for (int sector = DTChamberId::minSectorId; sector <= DTChamberId::maxSectorId; ++sector) {
          for (int superlayer = DTChamberId::minSuperLayerId; superlayer <= DTChamberId::maxSuperLayerId;
               ++superlayer) {
            iF(DTSuperLayerId(wheel, station, sector, superlayer));
          }
        }
      }
    }
  }

  template <typename F>
  void loop_all_layers(F&& iF) {
    loop_all_superlayers([&iF](auto superLayer) {
      for (int layer = DTChamberId::minLayerId; layer <= DTChamberId::maxLayerId; ++layer) {
        iF(DTLayerId(superLayer, layer));
      }
    });
  }

}  // namespace

TEST_CASE("Test DTRangeMapAccessor", "[DTRangeMapAccessor]") {
  SECTION("DTSuperLayerIdComparator") {
    SECTION("compare DTSuperLayerIds") {
      loop_all_superlayers([](auto iLayer1) {
        loop_all_superlayers([iLayer1](auto iLayer2) {
          DTSuperLayerIdComparator cmp;
          if (iLayer1 < iLayer2) {
            REQUIRE(cmp(iLayer1, iLayer2));
            REQUIRE(not cmp(iLayer2, iLayer1));
          } else if (iLayer2 < iLayer1) {
            REQUIRE(not cmp(iLayer1, iLayer2));
            REQUIRE(cmp(iLayer2, iLayer1));
          } else {
            REQUIRE(not cmp(iLayer1, iLayer2));
          }
        });
      });
    }
    SECTION("compare DTSuperLayerId to DTLayerId") {
      //NOTE: the DTSuperLayerIdComparator only works for this case
      // because the comparitor takes its arguments by value, which
      // means a new DTSuperLayerId must be created from the DTLayerId via
      // the call to DTSuperLayerId(DTSuperLayerId const&) and that constructor
      // explicitly masks out all the extra values coming from DTLayerId.

      loop_all_layers([](auto iLayer1) {
        loop_all_superlayers([iLayer1](auto iLayer2) {
          //                    std::cout <<"1 "<<iLayer1<<" 2 "<<iLayer2<<std::endl;
          DTSuperLayerIdComparator cmp;
          if (iLayer1 < iLayer2) {
            REQUIRE(cmp(iLayer1, iLayer2));
            REQUIRE(not cmp(iLayer2, iLayer1));
          } else if (iLayer2 < iLayer1) {
            REQUIRE(not cmp(iLayer1, iLayer2));
            //REQUIRE(cmp(iLayer2, iLayer1)); //cmp may think they are equal
          } else {
            REQUIRE(not cmp(iLayer1, iLayer2));
          }
        });
      });
    }
    SECTION("find all DTLayerId with same DTSuperLayerId") {
      std::vector<DTLayerId> ids;
      loop_all_layers([&ids](auto iID) { ids.emplace_back(iID); });
      std::sort(ids.begin(), ids.end());

      loop_all_superlayers([&ids](auto iSuper) {
        DTSuperLayerIdComparator cmp;
        auto range = std::equal_range(ids.begin(), ids.end(), iSuper, cmp);
        REQUIRE(range.second - range.first == DTChamberId::maxLayerId - DTChamberId::minLayerId + 1);
      });
    }
  }
}
