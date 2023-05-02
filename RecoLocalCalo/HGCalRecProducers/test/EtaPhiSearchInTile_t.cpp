#include <cassert>
#include <cmath>
#include <iostream>

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

int countEntries(const HGCalScintillatorLayerTiles &t, const std::array<int, 4> &limits) {
  int entries = 0;
  for (int e = limits[0]; e <= limits[1]; ++e) {
    for (int p = limits[2]; p <= limits[3]; ++p) {
      int phi = (p % HGCalScintillatorLayerTiles::type::nRows);
      auto global_bin = t.getGlobalBinByBin(e, phi);
      entries += t[global_bin].size();
    }
  }
  return entries;
}

TEST_CASE("Check the correct behaviour of searchBoxEtaPhi", "searchBoxEtaPhi") {
  using T = HGCalScintillatorLayerTiles::type;
  auto constexpr phiBins = T::nRows;
  auto constexpr etaBins = T::nColumns;
  auto constexpr phi_bin_width = 2. * M_PI / phiBins;
  auto constexpr eta_bin_width = (T::maxDim1 - T::minDim1) / etaBins;
  auto constexpr phi_transition_left = M_PI - 3. * phi_bin_width / 2.;
  auto constexpr phi_transition_right = M_PI + 3. * phi_bin_width / 2.;
  auto constexpr phi_transition_right2 = -M_PI + 5. * phi_bin_width / 2.;
  auto constexpr phi_transition_right3 = M_PI + 5. * phi_bin_width / 2.;
  unsigned int constexpr entries_left = 11;
  unsigned int constexpr entries_right = 7;
  float constexpr eta = 2.0f;
  float constexpr eta_neg = -2.4f;

  // Force filling using eta/phi vectors
  std::vector<bool> isSilicon(entries_left + entries_right, false);
  std::vector<float> dummy(entries_left + entries_right, 0.);
  std::vector<float> etas(entries_left + entries_right, eta);
  std::vector<float> etas_neg(entries_left + entries_right, eta_neg);
  std::vector<float> phis_left(entries_left, phi_transition_left);
  std::vector<float> phis_right(entries_right, phi_transition_right);
  std::vector<float> phis_right2(entries_right, phi_transition_right2);
  std::vector<float> phis;
  std::vector<float> phis2;
  phis.reserve(entries_left + entries_right);
  phis.insert(phis.end(), phis_left.begin(), phis_left.end());
  phis.insert(phis.end(), phis_right.begin(), phis_right.end());
  phis2.reserve(entries_left + entries_right);
  phis2.insert(phis2.end(), phis_left.begin(), phis_left.end());
  phis2.insert(phis2.end(), phis_right2.begin(), phis_right2.end());

  HGCalScintillatorLayerTiles t, t2, t_neg;
  t.fill(etas, phis);
  t2.fill(etas, phis2);
  t_neg.fill(etas_neg, phis2);

  std::cout << "Testing a Tile with " << phiBins << " phi bins with binwidth: " << phi_bin_width << " at pi transition"
            << std::endl;
  std::cout << "Testing a Tile with " << etaBins << " eta bins with binwidth: " << eta_bin_width << std::endl;

  std::cout << "-M_PI bin: " << t.mPiPhiBin << " M_PI bin: " << t.pPiPhiBin << std::endl;
  std::cout << "Filling positive eta value: " << eta << " at bin: " << t.getDim1Bin(eta) << std::endl;
  std::cout << "Filling negative eta value: " << eta_neg << " at bin: " << t.getDim1Bin(eta_neg) << std::endl;
  std::cout << "Filling phi value: " << phi_transition_left << " at left-pi bin: " << t.getDim2Bin(phi_transition_left)
            << std::endl;
  std::cout << "Filling phi value: " << phi_transition_right
            << " at right-pi bin: " << t.getDim2Bin(phi_transition_right) << std::endl;
  std::cout << "Filling phi value: " << phi_transition_right2
            << " at right-pi bin: " << t.getDim2Bin(phi_transition_right2) << std::endl;

  SECTION("Phi bins from positive and negative pi") {
    REQUIRE(t.getDim2Bin(phi_transition_right2) == t.getDim2Bin(phi_transition_right3));
  }

  SECTION("Symmetric case around pi") {
    auto limits = t.searchBox(1.95, 2.05, phi_transition_left, phi_transition_right);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Asymmetric case around pi, negative") {
    auto limits = t2.searchBox(1.95, 2.05, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t2, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Asymmetric case around pi, positive") {
    auto limits = t2.searchBox(1.95, 2.05, phi_transition_left, phi_transition_right3);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t2, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Correct all negative eta searchRange") {
    auto limits = t_neg.searchBox(-2.45, -2.35, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t_neg, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Mixed signs (neg,pos) eta with no entries searchRange") {
    auto limits = t.searchBox(-2.05, 1.95, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }

  SECTION("Mixed signs (neg,pos) eta with all entries searchRange") {
    auto limits = t.searchBox(-2.05, 2.05, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Mixed signs (neg,pos) eta with all entries negative eta searchRange") {
    auto limits = t_neg.searchBox(-2.45, 2.05, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t_neg, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Mixed signs (neg,pos) eta with all entries overflow eta searchRange") {
    auto limits = t_neg.searchBox(-5., 5., phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t_neg, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Wrong mixed signs (pos,neg) eta searchRange") {
    auto limits = t.searchBox(2.05, -1.95, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }

  SECTION("Wrong all positive eta searchRange") {
    auto limits = t.searchBox(2.05, 1.95, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }

  SECTION("Wrong all negative eta searchRange") {
    auto limits = t.searchBox(-1.95, -2.05, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }
}
