#ifndef __L1Trigger_L1THGCal_HGCalStage2ClusterDistribution_h__
#define __L1Trigger_L1THGCal_HGCalStage2ClusterDistribution_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"

class HGCalStage2ClusterDistribution {
public:
  HGCalStage2ClusterDistribution(const edm::ParameterSet& conf);

  HGCalTriggerGeometryBase::geom_set getStage2FPGAs(const unsigned stage1_fpga,
                                                    const HGCalTriggerGeometryBase::geom_set& stage2_fpgas,
                                                    const edm::Ptr<l1t::HGCalCluster>& tc_ptr) const;
  unsigned phiBin(unsigned roverzbin, double phi) const;
  double rotatedphi(double x, double y, double z, int sector) const;

private:
  double roz_min_ = 0.;
  double roz_max_ = 0.;
  unsigned roz_bins_ = 42;
  std::vector<double> phi_edges_;
  double roz_bin_size_ = 0.;
};

#endif
