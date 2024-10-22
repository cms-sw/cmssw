#ifndef __L1Trigger_L1THGCal_HGCalStage1TruncationConfig_SA_h__
#define __L1Trigger_L1THGCal_HGCalStage1TruncationConfig_SA_h__

#include <vector>
#include <cstdint>  // uint32_t

namespace l1thgcfirmware {

  class Stage1TruncationConfig {
  public:
    Stage1TruncationConfig(const bool do_truncate,
                           const double roz_min,
                           const double roz_max,
                           const unsigned roz_bins,
                           const std::vector<unsigned>& max_tcs_per_bins,
                           const std::vector<double>& phi_edges)
        : do_truncate_(do_truncate),
          roz_min_(roz_min),
          roz_max_(roz_max),
          roz_bins_(roz_bins),
          max_tcs_per_bins_(max_tcs_per_bins),
          phi_edges_(phi_edges) {}

    void setParameters(const bool do_truncate,
                       const double roz_min,
                       const double roz_max,
                       const unsigned roz_bins,
                       const std::vector<unsigned>& max_tcs_per_bins,
                       const std::vector<double>& phi_edges) {
      do_truncate_ = do_truncate;
      roz_min_ = roz_min;
      roz_max_ = roz_max;
      roz_bins_ = roz_bins;
      max_tcs_per_bins_ = max_tcs_per_bins;
      phi_edges_ = phi_edges;
    }

    void setParameters(const Stage1TruncationConfig& newConfig) {
      setParameters(newConfig.doTruncate(),
                    newConfig.rozMin(),
                    newConfig.rozMax(),
                    newConfig.rozBins(),
                    newConfig.maxTcsPerBin(),
                    newConfig.phiEdges());
    }

    void setSector120(const unsigned sector) { sector120_ = sector; }
    void setFPGAID(const uint32_t fpga_id) { fpga_id_ = fpga_id; }

    bool doTruncate() const { return do_truncate_; }
    double rozMin() const { return roz_min_; }
    double rozMax() const { return roz_max_; }
    unsigned rozBins() const { return roz_bins_; }
    const std::vector<unsigned>& maxTcsPerBin() const { return max_tcs_per_bins_; }
    const std::vector<double>& phiEdges() const { return phi_edges_; }
    unsigned phiSector() const { return sector120_; }
    uint32_t fpgaID() const { return fpga_id_; }

  private:
    bool do_truncate_;
    double roz_min_;
    double roz_max_;
    unsigned roz_bins_;
    std::vector<unsigned> max_tcs_per_bins_;
    std::vector<double> phi_edges_;
    unsigned sector120_;
    uint32_t fpga_id_;
  };

}  // namespace l1thgcfirmware

#endif
