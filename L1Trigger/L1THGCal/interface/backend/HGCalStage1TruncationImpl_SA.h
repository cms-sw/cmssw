#ifndef __L1Trigger_L1THGCal_HGCalStage1TruncationImpl_h__
#define __L1Trigger_L1THGCal_HGCalStage1TruncationImpl_h__

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerCell_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalStage1TruncationConfig_SA.h"

#include <vector>
#include <cstdint>        // uint32_t, unsigned
#include <unordered_map>  // std::unordered_map
#include <algorithm>      // std::sort

class HGCalStage1TruncationImplSA {
public:
  HGCalStage1TruncationImplSA() = default;
  ~HGCalStage1TruncationImplSA() = default;

  void runAlgorithm() const;

  unsigned run(const l1thgcfirmware::HGCalTriggerCellSACollection& tcs_in,
               const l1thgcfirmware::Stage1TruncationConfig& theConf,
               l1thgcfirmware::HGCalTriggerCellSACollection& tcs_out) const;

  int phiBin(unsigned roverzbin, double phi, const std::vector<double>& phiedges) const;
  double rotatedphi(double x, double y, double z, int sector) const;
  unsigned rozBin(double roverz, double rozmin, double rozmax, unsigned rozbins) const;

private:
  static constexpr unsigned offset_roz_ = 1;
  static constexpr unsigned mask_roz_ = 0x3f;  // 6 bits, max 64 bins
  static constexpr unsigned mask_phi_ = 1;

  bool do_truncate_;
  double roz_min_ = 0.;
  double roz_max_ = 0.;
  unsigned roz_bins_ = 42;
  std::vector<unsigned> max_tcs_per_bin_;
  std::vector<double> phi_edges_;

  uint32_t packBin(unsigned roverzbin, unsigned phibin) const;
  void unpackBin(unsigned packedbin, unsigned& roverzbin, unsigned& phibin) const;
};

#endif
