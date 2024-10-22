#include <cassert>
#include <cmath>
#include <iostream>

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using namespace ticl;

int countEntries(const TICLLayerTile &t, const std::array<int, 4> &limits) {
  int entries = 0;
  for (int e = limits[0]; e <= limits[1]; ++e) {
    for (int p = limits[2]; p <= limits[3]; ++p) {
      int phi = (p % TICLLayerTile::type::nPhiBins);
      auto global_bin = t.globalBin(e, phi);
      entries += t[global_bin].size();
    }
  }
  return entries;
}

TEST_CASE("Check the correct phi wrapping", "searchBoxEtaPhi") {
  auto constexpr phiBins = TICLLayerTile::type::nPhiBins;
  auto constexpr phi_bin_width = 2. * M_PI / phiBins;
  auto constexpr phi_transition_left = M_PI - phi_bin_width / 2.;
  auto constexpr phi_transition_right = M_PI + phi_bin_width / 2.;
  auto constexpr phi_transition_right2 = -M_PI + 3. * phi_bin_width / 2.;
  auto constexpr phi_transition_right3 = M_PI + 3. * phi_bin_width / 2.;
  unsigned int constexpr entries_left = 11;
  unsigned int constexpr entries_right = 7;
  float constexpr eta = 2.0f;
  float constexpr eta_neg = -2.4f;

  TICLLayerTile t, t2, t_neg;
  std::cout << "Testing a Tile with " << phiBins << " bins with binwidth: " << phi_bin_width << " at bin transition"
            << std::endl;
  std::cout << "Filling left-pi bin: " << t.phiBin(phi_transition_left) << std::endl;
  std::cout << "Filling right-pi bin: " << t.phiBin(phi_transition_right) << std::endl;
  std::cout << "Filling right2-pi bin: " << t.phiBin(phi_transition_right2) << std::endl;
  std::cout << "Filling right3-pi bin: " << t.phiBin(phi_transition_right3) << std::endl;

  for (unsigned int i = 0; i < entries_left; ++i) {
    t.fill(eta, phi_transition_left, i);
    t2.fill(eta, phi_transition_left, i);
    t_neg.fill(eta_neg, phi_transition_left, i);
  }

  for (unsigned int i = 0; i < entries_right; ++i) {
    t.fill(eta, phi_transition_right, i);
    t2.fill(eta, phi_transition_right2, i);
    t_neg.fill(eta_neg, phi_transition_right, i);
  }

  SECTION("Phi bins from positive and negative pi") {
    REQUIRE(t.phiBin(phi_transition_right2) == t.phiBin(phi_transition_right3));
  }

  SECTION("Symmetric case around pi") {
    auto limits = t.searchBoxEtaPhi(1.95, 2.05, phi_transition_left, phi_transition_right);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Asymmetric case around pi, negative") {
    auto limits = t2.searchBoxEtaPhi(1.95, 2.05, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t2, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Asymmetric case around pi, positive") {
    auto limits = t2.searchBoxEtaPhi(1.95, 2.05, phi_transition_left, phi_transition_right3);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t2, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Correct all negative eta searchRange") {
    auto limits = t_neg.searchBoxEtaPhi(-2.45, -2.35, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t_neg, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Correct all positive overflow eta searchRange") {
    auto limits = t.searchBoxEtaPhi(0., 5., phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] <= limits[1]);
    REQUIRE(limits[2] <= limits[3]);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == entries_left + entries_right);
  }

  SECTION("Wrong mixed signs (neg,pos) eta searchRange") {
    auto limits = t.searchBoxEtaPhi(-2.05, 1.95, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }

  SECTION("Wrong mixed signs (pos,neg) eta searchRange") {
    auto limits = t.searchBoxEtaPhi(2.05, -1.95, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }

  SECTION("Wrong all positive eta searchRange") {
    auto limits = t.searchBoxEtaPhi(2.05, 1.95, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }

  SECTION("Wrong all negative eta searchRange") {
    auto limits = t.searchBoxEtaPhi(-1.95, -2.05, phi_transition_left, phi_transition_right2);

    std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
    REQUIRE(limits[0] == limits[1]);
    REQUIRE(limits[2] == limits[3]);
    REQUIRE(limits[0] == 0);
    REQUIRE(limits[2] == 0);

    auto entries = countEntries(t, limits);
    REQUIRE(entries == 0);
  }
}
