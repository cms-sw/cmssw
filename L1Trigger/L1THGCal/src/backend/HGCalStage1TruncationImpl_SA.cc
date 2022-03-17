#include "L1Trigger/L1THGCal/interface/backend/HGCalStage1TruncationImpl_SA.h"
#include <cmath>

unsigned HGCalStage1TruncationImplSA::run(const l1thgcfirmware::HGCalTriggerCellSACollection& tcs_in,
                                          const l1thgcfirmware::Stage1TruncationConfig& theConf,
                                          l1thgcfirmware::HGCalTriggerCellSACollection& tcs_out) const {
  unsigned sector120 = theConf.phiSector();
  std::unordered_map<unsigned, l1thgcfirmware::HGCalTriggerCellSACollection> tcs_per_bin;

  // configuation:
  bool do_truncate = theConf.doTruncate();
  double rozmin = theConf.rozMin();
  double rozmax = theConf.rozMax();
  unsigned rozbins = theConf.rozBins();
  const std::vector<unsigned>& maxtcsperbin = theConf.maxTcsPerBin();
  const std::vector<double>& phiedges = theConf.phiEdges();

  // group TCs per (r/z, phi) bins
  for (const auto& tc : tcs_in) {
    double x = tc.x();
    double y = tc.y();
    double z = tc.z();
    double roverz = std::sqrt(x * x + y * y) / std::abs(z);
    unsigned roverzbin = rozBin(roverz, rozmin, rozmax, rozbins);

    double phi = rotatedphi(x, y, z, sector120);
    int phibin = phiBin(roverzbin, phi, phiedges);
    if (phibin < 0)
      return 1;
    unsigned packed_bin = packBin(roverzbin, phibin);

    tcs_per_bin[packed_bin].push_back(tc);
  }
  // apply sorting and trunction in each (r/z, phi) bin
  for (auto& bin_tcs : tcs_per_bin) {
    std::sort(bin_tcs.second.begin(),
              bin_tcs.second.end(),
              [](const l1thgcfirmware::HGCalTriggerCell& a, const l1thgcfirmware::HGCalTriggerCell& b) -> bool {
                return a.mipPt() > b.mipPt();
              });

    unsigned roverzbin = 0;
    unsigned phibin = 0;
    unpackBin(bin_tcs.first, roverzbin, phibin);
    if (roverzbin >= maxtcsperbin.size())
      return 1;

    unsigned max_tc = maxtcsperbin[roverzbin];
    if (do_truncate && bin_tcs.second.size() > max_tc) {
      bin_tcs.second.resize(max_tc);
    }

    for (const auto& tc : bin_tcs.second) {
      tcs_out.push_back(tc);
    }
  }

  return 0;
}

unsigned HGCalStage1TruncationImplSA::packBin(unsigned roverzbin, unsigned phibin) const {
  unsigned packed_bin = 0;
  packed_bin |= ((roverzbin & mask_roz_) << offset_roz_);
  packed_bin |= (phibin & mask_phi_);
  return packed_bin;
}

void HGCalStage1TruncationImplSA::unpackBin(unsigned packedbin, unsigned& roverzbin, unsigned& phibin) const {
  roverzbin = ((packedbin >> offset_roz_) & mask_roz_);
  phibin = (packedbin & mask_phi_);
}

int HGCalStage1TruncationImplSA::phiBin(unsigned roverzbin, double phi, const std::vector<double>& phiedges) const {
  int phi_bin = 0;
  if (roverzbin >= phiedges.size())
    return -1;
  double phi_edge = phiedges[roverzbin];
  if (phi > phi_edge)
    phi_bin = 1;
  return phi_bin;
}

double HGCalStage1TruncationImplSA::rotatedphi(double x, double y, double z, int sector) const {
  if (z > 0)
    x = -x;
  double phi = std::atan2(y, x);

  if (sector == 1) {
    if (phi < M_PI and phi > 0)
      phi = phi - (2. * M_PI / 3.);
    else
      phi = phi + (4. * M_PI / 3.);
  } else if (sector == 2) {
    phi = phi + (2. * M_PI / 3.);
  }
  return phi;
}

unsigned HGCalStage1TruncationImplSA::rozBin(double roverz, double rozmin, double rozmax, unsigned rozbins) const {
  constexpr double margin = 1.001;
  double roz_bin_size = (rozbins > 0 ? (rozmax - rozmin) * margin / double(rozbins) : 0.);
  unsigned roverzbin = 0;
  if (roz_bin_size > 0.) {
    roverz -= rozmin;
    roverz = std::clamp(roverz, 0., rozmax - rozmin);
    roverzbin = unsigned(roverz / roz_bin_size);
  }

  return roverzbin;
}
