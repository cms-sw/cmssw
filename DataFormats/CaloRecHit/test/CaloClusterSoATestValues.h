#ifndef DataFormats_CaloRecHit_test_CaloClusterSoATestValues_h
#define DataFormats_CaloRecHit_test_CaloClusterSoATestValues_h

#include <Eigen/Core>

#include <cstdint>
#include <string_view>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "FWCore/Utilities/interface/Exception.h"

// Known values for the CaloCluster SoA collection of the unit tests: fill() writes them, check() throws if one differs.
namespace caloclustertest {

  // The value of field k of element i: exact in float for the sizes of the tests.
  inline float value(int i, int k) { return 100.f * i + k + 0.25f; }

  inline void require(bool condition, std::string_view what, int i) {
    if (not condition)
      throw cms::Exception("TestFailure") << what << " differs at index " << i;
  }

  inline void requireSize(int32_t size, int32_t expected, std::string_view what) {
    if (size != expected)
      throw cms::Exception("TestFailure") << what << " is " << size << ", expected " << expected;
  }

  inline void fill(reco::CaloClusterSoAView view) {
    const int n = view.position().metadata().size();
    for (int i = 0; i < n; ++i) {
      view.position()[i].x() = value(i, 0);
      view.position()[i].y() = value(i, 1);
      view.position()[i].z() = value(i, 2);
      view.position()[i].layer() = i % 47;
      view.position()[i].cells() = 3 * i;
      view.energy()[i].energy() = value(i, 3);
      view.energy()[i].correctedEnergy() = value(i, 4);
      view.energy()[i].correctedEnergyUncertainty() = value(i, 5);
      view.indexes()[i].algoID() = reco::CaloCluster::hgcal_em;
      view.indexes()[i].seedID() = 0x10000000u + i;
      view.indexes()[i].flags() = 7u * i;
      view.timing()[i].time() = value(i, 6);
      view.timing()[i].timeError() = value(i, 7);
    }
  }

  inline void check(reco::CaloClusterSoAConstView view, int32_t n) {
    requireSize(view.position().metadata().size(), n, "position size");
    requireSize(view.energy().metadata().size(), n, "energy size");
    requireSize(view.indexes().metadata().size(), n, "indexes size");
    requireSize(view.timing().metadata().size(), n, "timing size");
    for (int i = 0; i < n; ++i) {
      require(view.position()[i].x() == value(i, 0), "x", i);
      require(view.position()[i].y() == value(i, 1), "y", i);
      require(view.position()[i].z() == value(i, 2), "z", i);
      require(view.position()[i].layer() == i % 47, "layer", i);
      require(view.position()[i].cells() == 3 * i, "cells", i);
      require(view.energy()[i].energy() == value(i, 3), "energy", i);
      require(view.energy()[i].correctedEnergy() == value(i, 4), "correctedEnergy", i);
      require(view.energy()[i].correctedEnergyUncertainty() == value(i, 5), "correctedEnergyUncertainty", i);
      require(view.indexes()[i].algoID() == reco::CaloCluster::hgcal_em, "algoID", i);
      require(view.indexes()[i].seedID() == 0x10000000u + i, "seedID", i);
      require(view.indexes()[i].flags() == 7u * i, "flags", i);
      require(view.timing()[i].time() == value(i, 6), "time", i);
      require(view.timing()[i].timeError() == value(i, 7), "timeError", i);
    }
  }

}  // namespace caloclustertest

#endif  // DataFormats_CaloRecHit_test_CaloClusterSoATestValues_h
