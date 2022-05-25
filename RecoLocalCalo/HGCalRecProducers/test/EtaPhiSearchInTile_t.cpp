#include <cassert>
#include <cmath>
#include <iostream>

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"

void runTest(HGCalLayerTiles const& t, int expected, float etaMin, float etaMax, float phiMin, float phiMax) {
  auto limits = t.searchBoxEtaPhi(etaMin, etaMax, phiMin, phiMax);

  std::cout << "Limits are: " << limits[0] << " " << limits[1] << " " << limits[2] << " " << limits[3] << std::endl;
  assert(limits[0] <= limits[1]);
  assert(limits[2] <= limits[3]);

  int entries = 0;
  for (int e = limits[0]; e <= limits[1]; ++e) {
    for (int p = limits[2]; p <= limits[3]; ++p) {
      int phi = (p % HGCalLayerTiles::type::nRowsPhi);
      auto global_bin = t.getGlobalBinByBinEtaPhi(e, phi);
      entries += t[global_bin].size();
    }
  }

  std::cout << "Found " << entries << " entries, expected " << expected << std::endl;
  assert(entries == expected);
}

int main(int argc, char* argv[]) {
  auto constexpr phiBins = HGCalLayerTiles::type::nRowsPhi;
  auto constexpr phi_bin_width = 2. * M_PI / phiBins;
  auto constexpr phi_transition_left = M_PI - 3. * phi_bin_width;
  auto constexpr phi_transition_right = M_PI + 3. * phi_bin_width;
  auto constexpr phi_transition_right2 = -M_PI + 5. * phi_bin_width;
  unsigned int constexpr entries_left = 11;
  unsigned int constexpr entries_right = 7;
  float constexpr eta = 2.0;

  // Force filling using eta/phi vectors
  std::vector<bool> isSilicon(entries_left + entries_right, false);
  std::vector<float> dummy(entries_left + entries_right, 0.);
  std::vector<float> etas(entries_left + entries_right, eta);
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

  HGCalLayerTiles t, t2;
  t.fill(dummy, dummy, etas, phis, isSilicon);
  t2.fill(dummy, dummy, etas, phis2, isSilicon);

  std::cout << "Testing a Tile with " << phiBins << " bins with binwidth: " << phi_bin_width << " at pi transition"
            << std::endl;
  std::cout << "-M_PI bin: " << t.mPiPhiBin << " M_PI bin: " << t.pPiPhiBin << std::endl;
  std::cout << "Filling phi value: " << phi_transition_left << " at left-pi bin: " << t.getPhiBin(phi_transition_left)
            << std::endl;
  std::cout << "Filling phi value: " << phi_transition_right
            << " at right-pi bin: " << t.getPhiBin(phi_transition_right) << std::endl;
  std::cout << "Filling phi value: " << phi_transition_right2
            << " at right-pi bin: " << t.getPhiBin(phi_transition_right2) << std::endl;

  runTest(t, entries_right + entries_left, 1.95, 2.05, phi_transition_left, phi_transition_right);
  runTest(t2, entries_right + entries_left, 1.95, 2.05, phi_transition_left, phi_transition_right2);

  return 0;
}
