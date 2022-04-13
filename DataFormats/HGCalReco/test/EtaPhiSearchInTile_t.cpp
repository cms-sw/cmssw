#include <cassert>
#include <cmath>
#include <iostream>

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

using namespace ticl;

void runTest(TICLLayerTile const& t, int expected, float etaMin, float etaMax, float phiMin, float phiMax) {
  auto limits = t.searchBoxEtaPhi(etaMin, etaMax, phiMin, phiMax);

  std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
  assert(limits[0] <= limits[1]);
  assert(limits[2] <= limits[3]);

  int entries = 0;
  for (int e = limits[0]; e <= limits[1]; ++e) {
    for (int p = limits[2]; p <= limits[3]; ++p) {
      int phi = (p % TICLLayerTile::type::nPhiBins);
      auto global_bin = t.globalBin(e, phi);
      entries += t[global_bin].size();
    }
  }

  std::cout << "Found " << entries << " entries, expected " << expected << std::endl;
  assert(entries == expected);
}

int main(int argc, char* argv[]) {
  auto constexpr phiBins = TICLLayerTile::type::nPhiBins;
  auto constexpr phi_bin_width = 2. * M_PI / phiBins;
  auto constexpr phi_transition_left = M_PI - phi_bin_width;
  auto constexpr phi_transition_right = M_PI + phi_bin_width;
  auto constexpr phi_transition_right2 = -M_PI + 3. * phi_bin_width;
  unsigned int constexpr entries_left = 11;
  unsigned int constexpr entries_right = 7;
  float constexpr eta = 2.0;

  TICLLayerTile t, t2;
  std::cout << "Testing a Tile with " << phiBins << " bins with binwidth: " << phi_bin_width << " at bin transition"
            << std::endl;
  std::cout << "Filling left-pi bin: " << t.phiBin(phi_transition_left) << std::endl;
  std::cout << "Filling right-pi bin: " << t.phiBin(phi_transition_right) << std::endl;
  std::cout << "Filling right2-pi bin: " << t.phiBin(phi_transition_right2) << std::endl;

  for (unsigned int i = 0; i < entries_left; ++i) {
    t.fill(eta, phi_transition_left, i);
    t2.fill(eta, phi_transition_left, i);
  }

  for (unsigned int i = 0; i < entries_right; ++i) {
    t.fill(eta, phi_transition_right, i);
    t2.fill(eta, phi_transition_right2, i);
  }

  runTest(t, entries_left + entries_right, 1.95, 2.05, phi_transition_left, phi_transition_right);
  runTest(t2, entries_left + entries_right, 1.95, 2.05, phi_transition_left, phi_transition_right2);

  return 0;
}
