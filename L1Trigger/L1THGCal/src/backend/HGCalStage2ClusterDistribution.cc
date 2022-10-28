#include "L1Trigger/L1THGCal/interface/backend/HGCalStage2ClusterDistribution.h"
#include "DataFormats/Common/interface/PtrVector.h"

//class constructor
HGCalStage2ClusterDistribution::HGCalStage2ClusterDistribution(const edm::ParameterSet& conf)
    : roz_min_(conf.getParameter<double>("rozMin")),
      roz_max_(conf.getParameter<double>("rozMax")),
      roz_bins_(conf.getParameter<unsigned>("rozBins")),
      phi_edges_(conf.getParameter<std::vector<double>>("phiSectorEdges")) {
  if (phi_edges_.size() != roz_bins_)
    throw cms::Exception("HGCalStage2ClusterDistribution::BadConfig")
        << "Inconsistent sizes of phiSectorEdges and rozBins";

  constexpr double margin = 1.001;
  roz_bin_size_ = (roz_bins_ > 0 ? (roz_max_ - roz_min_) * margin / double(roz_bins_) : 0.);
}

HGCalTriggerGeometryBase::geom_set HGCalStage2ClusterDistribution::getStage2FPGAs(
    const unsigned stage1_fpga,
    const HGCalTriggerGeometryBase::geom_set& stage2_fpgas,
    const edm::Ptr<l1t::HGCalCluster>& tc_ptr) const {
  HGCalTriggerBackendDetId stage1_fpga_id(stage1_fpga);
  int sector120 = stage1_fpga_id.sector();

  const GlobalPoint& position = tc_ptr->position();
  double x = position.x();
  double y = position.y();
  double z = position.z();
  double roverz = std::sqrt(x * x + y * y) / std::abs(z);
  roverz = (roverz < roz_min_ ? roz_min_ : roverz);
  roverz = (roverz > roz_max_ ? roz_max_ : roverz);
  unsigned roverzbin = (roz_bin_size_ > 0. ? unsigned((roverz - roz_min_) / roz_bin_size_) : 0);
  double phi = rotatedphi(x, y, z, sector120);
  unsigned phibin = phiBin(roverzbin, phi);

  HGCalTriggerGeometryBase::geom_set output_fpgas;

  for (const auto& fpga : stage2_fpgas) {
    if (phibin == 0 || sector120 == HGCalTriggerBackendDetId(fpga).sector()) {
      output_fpgas.emplace(fpga);
    }
  }

  return output_fpgas;
}

unsigned HGCalStage2ClusterDistribution::phiBin(unsigned roverzbin, double phi) const {
  unsigned phi_bin = 0;
  if (roverzbin >= phi_edges_.size())
    throw cms::Exception("HGCalStage1TruncationImpl::OutOfRange") << "roverzbin index " << roverzbin << "out of range";
  double phi_edge = phi_edges_[roverzbin];
  if (phi > phi_edge)
    phi_bin = 1;
  return phi_bin;
}

double HGCalStage2ClusterDistribution::rotatedphi(double x, double y, double z, int sector) const {
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
