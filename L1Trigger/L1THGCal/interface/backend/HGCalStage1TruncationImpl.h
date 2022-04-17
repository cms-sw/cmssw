#ifndef __L1Trigger_L1THGCal_HGCalStage1TruncationImpl_h__
#define __L1Trigger_L1THGCal_HGCalStage1TruncationImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include <vector>

class HGCalStage1TruncationImpl {
public:
  HGCalStage1TruncationImpl(const edm::ParameterSet& conf);

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

  void run(uint32_t fpga_id,
           const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& tcs_in,
           std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& tcs_out);

private:
  HGCalTriggerTools triggerTools_;

  static constexpr unsigned offset_roz_ = 1;
  static constexpr unsigned mask_roz_ = 0x3f;  // 6 bits, max 64 bins
  static constexpr unsigned mask_phi_ = 1;

  bool do_truncate_;
  double roz_min_ = 0.;
  double roz_max_ = 0.;
  unsigned roz_bins_ = 42;
  std::vector<unsigned> max_tcs_per_bin_;
  std::vector<double> phi_edges_;
  double roz_bin_size_ = 0.;

  uint32_t packBin(unsigned roverzbin, unsigned phibin) const;
  void unpackBin(unsigned packedbin, unsigned& roverzbin, unsigned& phibin) const;
  unsigned phiBin(unsigned roverzbin, double phi) const;
  double rotatedphi(double x, double y, double z, int sector) const;
};

#endif
