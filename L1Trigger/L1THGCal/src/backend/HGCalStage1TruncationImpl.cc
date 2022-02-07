#include "L1Trigger/L1THGCal/interface/backend/HGCalStage1TruncationImpl.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"
#include <cmath>

HGCalStage1TruncationImpl::HGCalStage1TruncationImpl(const edm::ParameterSet& conf)
    : do_truncate_(conf.getParameter<bool>("doTruncation")),
      roz_min_(conf.getParameter<double>("rozMin")),
      roz_max_(conf.getParameter<double>("rozMax")),
      roz_bins_(conf.getParameter<unsigned>("rozBins")),
      max_tcs_per_bin_(conf.getParameter<std::vector<unsigned>>("maxTcsPerBin")),
      phi_edges_(conf.getParameter<std::vector<double>>("phiSectorEdges")) {
  if (max_tcs_per_bin_.size() != roz_bins_)
    throw cms::Exception("HGCalStage1TruncationImpl::BadConfig") << "Inconsistent sizes of maxTcsPerBin and rozBins";
  if (phi_edges_.size() != roz_bins_)
    throw cms::Exception("HGCalStage1TruncationImpl::BadConfig") << "Inconsistent sizes of phiSectorEdges and rozBins";

  constexpr double margin = 1.001;
  roz_bin_size_ = (roz_bins_ > 0 ? (roz_max_ - roz_min_) * margin / double(roz_bins_) : 0.);
}

void HGCalStage1TruncationImpl::run(uint32_t fpga_id,
                                    const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& tcs_in,
                                    std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& tcs_out) {
  unsigned sector120 = HGCalTriggerBackendDetId(fpga_id).sector();
  std::unordered_map<unsigned, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>> tcs_per_bin;

  // group TCs per (r/z, phi) bins
  for (const auto& tc : tcs_in) {
    const GlobalPoint& position = tc->position();
    double x = position.x();
    double y = position.y();
    double z = position.z();
    unsigned roverzbin = 0;
    if (roz_bin_size_ > 0.) {
      double roverz = std::sqrt(x * x + y * y) / std::abs(z) - roz_min_;
      roverz = std::clamp(roverz, 0., roz_max_ - roz_min_);
      roverzbin = unsigned(roverz / roz_bin_size_);
    }
    double phi = rotatedphi(x, y, z, sector120);
    unsigned phibin = phiBin(roverzbin, phi);
    unsigned packed_bin = packBin(roverzbin, phibin);

    tcs_per_bin[packed_bin].push_back(tc);
  }
  // apply sorting and trunction in each (r/z, phi) bin
  for (auto& bin_tcs : tcs_per_bin) {
    std::sort(bin_tcs.second.begin(),
              bin_tcs.second.end(),
              [](const edm::Ptr<l1t::HGCalTriggerCell>& a, const edm::Ptr<l1t::HGCalTriggerCell>& b) -> bool {
                return a->mipPt() > b->mipPt();
              });

    unsigned roverzbin = 0;
    unsigned phibin = 0;
    unpackBin(bin_tcs.first, roverzbin, phibin);
    if (roverzbin >= max_tcs_per_bin_.size())
      throw cms::Exception("HGCalStage1TruncationImpl::OutOfRange")
          << "roverzbin index " << roverzbin << "out of range";
    unsigned max_tc = max_tcs_per_bin_[roverzbin];
    if (do_truncate_ && bin_tcs.second.size() > max_tc) {
      bin_tcs.second.resize(max_tc);
    }
    for (const auto& tc : bin_tcs.second) {
      tcs_out.push_back(tc);
    }
  }
}

unsigned HGCalStage1TruncationImpl::packBin(unsigned roverzbin, unsigned phibin) const {
  unsigned packed_bin = 0;
  packed_bin |= ((roverzbin & mask_roz_) << offset_roz_);
  packed_bin |= (phibin & mask_phi_);
  return packed_bin;
}

void HGCalStage1TruncationImpl::unpackBin(unsigned packedbin, unsigned& roverzbin, unsigned& phibin) const {
  roverzbin = ((packedbin >> offset_roz_) & mask_roz_);
  phibin = (packedbin & mask_phi_);
}

unsigned HGCalStage1TruncationImpl::phiBin(unsigned roverzbin, double phi) const {
  unsigned phi_bin = 0;
  if (roverzbin >= phi_edges_.size())
    throw cms::Exception("HGCalStage1TruncationImpl::OutOfRange") << "roverzbin index " << roverzbin << "out of range";
  double phi_edge = phi_edges_[roverzbin];
  if (phi > phi_edge)
    phi_bin = 1;
  return phi_bin;
}

double HGCalStage1TruncationImpl::rotatedphi(double x, double y, double z, int sector) const {
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
