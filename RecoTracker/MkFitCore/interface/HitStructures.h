#ifndef RecoTracker_MkFitCore_interface_HitStructures_h
#define RecoTracker_MkFitCore_interface_HitStructures_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <algorithm>
#include <array>

namespace mkfit {

  class IterationParams;

  typedef std::pair<uint16_t, uint16_t> PhiBinInfo_t;

  typedef std::array<PhiBinInfo_t, Config::m_nphi> vecPhiBinInfo_t;

  typedef std::vector<vecPhiBinInfo_t> vecvecPhiBinInfo_t;

  typedef std::array<bool, Config::m_nphi> vecPhiBinDead_t;

  typedef std::vector<vecPhiBinDead_t> vecvecPhiBinDead_t;

  //==============================================================================
  //==============================================================================

  // Note: the same code is used for barrel and endcap. In barrel the longitudinal
  // bins are in Z and in endcap they are in R -- here this coordinate is called Q

  // When not defined, hits are accessed from the original hit vector and
  // only sort ranks are kept for proper access.
  //
  //#define COPY_SORTED_HITS

  class LayerOfHits {
  public:
    LayerOfHits() = default;

    ~LayerOfHits();

    // Setup and filling
    //-------------------
    void setupLayer(const LayerInfo& li);

    void reset() {}

    // Get in all hits from given hit-vec
    void suckInHits(const HitVec& hitv);

    // Get in all dead regions from given dead-vec
    void suckInDeads(const DeadVec& deadv);

    // Use external hit-vec and only use hits that are passed to me.
    void beginRegistrationOfHits(const HitVec& hitv);
    void registerHit(int idx);
    void endRegistrationOfHits(bool build_original_to_internal_map);

    int nHits() const { return m_n_hits; }

    // Bin access / queries
    //----------------------
    int qBin(float q) const { return (q - m_qmin) * m_fq; }

    int qBinChecked(float q) const { return std::clamp(qBin(q), 0, m_nq - 1); }

    // if you don't pass phi in (-pi, +pi), mask away the upper bits using m_phi_mask or use the Checked version.
    int phiBinFine(float phi) const { return std::floor(m_fphi_fine * (phi + Const::PI)); }
    int phiBin(float phi) const { return phiBinFine(phi) >> m_phi_bits_shift; }

    int phiBinChecked(float phi) const { return phiBin(phi) & m_phi_mask; }

    int phiMaskApply(int in) const { return in & m_phi_mask; }

    const vecPhiBinInfo_t& vecPhiBinInfo(float q) const { return m_phi_bin_infos[qBin(q)]; }

    const vecvecPhiBinInfo_t& phi_bin_infos() const { return m_phi_bin_infos; }
    const vecvecPhiBinDead_t& phi_bin_deads() const { return m_phi_bin_deads; }
    PhiBinInfo_t phi_bin_info(int qi, int pi) const { return m_phi_bin_infos[qi][pi]; }
    bool phi_bin_dead(int qi, int pi) const { return m_phi_bin_deads[qi][pi]; }

    float hit_q(int i) const { return m_hit_qs[i]; }
    float hit_phi(int i) const { return m_hit_phis[i]; }

    // Use this to map original indices to sorted internal ones. m_ext_idcs needs to be initialized.
    int getHitIndexFromOriginal(int i) const { return m_ext_idcs[i - m_min_ext_idx]; }
    // Use this to remap internal hit index to external one.
    int getOriginalHitIndex(int i) const { return m_hit_ranks[i]; }

#ifdef COPY_SORTED_HITS
    const Hit& refHit(int i) const { return m_hits[i]; }
    const Hit* hitArray() const { return m_hits; }
#else
    const Hit& refHit(int i) const { return (*m_ext_hits)[i]; }
    const Hit* hitArray() const { return m_ext_hits->data(); }
#endif

    // Left to document and demonstrate access to bin-info structures.
    // void  selectHitIndices(float q, float phi, float dq, float dphi, std::vector<int>& idcs, bool isForSeeding=false, bool dump=false);

    void printBins();

    // Geometry / LayerInfo accessors
    //--------------------------------

    const LayerInfo* layer_info() const { return m_layer_info; }
    int layer_id() const { return m_layer_info->layer_id(); }

    bool is_barrel() const { return m_is_barrel; }
    bool is_endcap() const { return !m_is_barrel; }

    bool is_within_z_limits(float z) const { return m_layer_info->is_within_z_limits(z); }
    bool is_within_r_limits(float r) const { return m_layer_info->is_within_r_limits(r); }

    WSR_Result is_within_z_sensitive_region(float z, float dz) const {
      return m_layer_info->is_within_z_sensitive_region(z, dz);
    }

    WSR_Result is_within_r_sensitive_region(float r, float dr) const {
      return m_layer_info->is_within_r_sensitive_region(r, dr);
    }

    bool is_stereo() const { return m_layer_info->is_stereo(); }
    bool is_pixel() const { return m_layer_info->is_pixel(); }
    int subdet() const { return m_layer_info->subdet(); }

  private:
    // Constants for phi-bin access / index manipulation.
    static constexpr float m_fphi = Config::m_nphi / Const::TwoPI;
    static constexpr int m_phi_mask = 0xff;
    static constexpr int m_phi_bits = 8;
    static constexpr float m_fphi_fine = 1024 / Const::TwoPI;
    static constexpr int m_phi_mask_fine = 0x3ff;
    static constexpr int m_phi_bits_fine = 10;  //can't be more than 16
    static constexpr int m_phi_bits_shift = m_phi_bits_fine - m_phi_bits;
    static constexpr int m_phi_fine_xmask = ~((1 << m_phi_bits_shift) - 1);

    void setup_bins(float qmin, float qmax, float dq);

    void empty_phi_bins(int q_bin, int phi_bin_1, int phi_bin_2, uint16_t hit_count) {
      for (int pb = phi_bin_1; pb < phi_bin_2; ++pb) {
        m_phi_bin_infos[q_bin][pb] = {hit_count, hit_count};
      }
    }

    void empty_q_bins(int q_bin_1, int q_bin_2, uint16_t hit_count) {
      for (int qb = q_bin_1; qb < q_bin_2; ++qb) {
        empty_phi_bins(qb, 0, Config::m_nphi, hit_count);
      }
    }

    void empty_phi_bins_dead(int q_bin, int phi_bin_1, int phi_bin_2) {
      for (int pb = phi_bin_1; pb < phi_bin_2; ++pb) {
        m_phi_bin_deads[q_bin][pb] = false;
      }
    }

    void empty_q_bins_dead(int q_bin_1, int q_bin_2) {
      for (int qb = q_bin_1; qb < q_bin_2; ++qb) {
        empty_phi_bins_dead(qb, 0, Config::m_nphi);
      }
    }

#ifdef COPY_SORTED_HITS
    void alloc_hits(int size);
    void free_hits()

        Hit* m_hits = nullptr;
    int m_capacity = 0;
#else
    const HitVec* m_ext_hits;
#endif
    unsigned int* m_hit_ranks = nullptr;  // allocated by IceSort via new []
    std::vector<int> m_ext_idcs;
    int m_min_ext_idx, m_max_ext_idx;
    int m_n_hits = 0;

    // Bin information for hits and dead regions
    vecvecPhiBinInfo_t m_phi_bin_infos;
    vecvecPhiBinDead_t m_phi_bin_deads;

    // Cached hit phi and q values to minimize Hit memory access
    std::vector<float> m_hit_phis;
    std::vector<float> m_hit_qs;

    // Geometry / q-binning constants - initialized in setupLayer()
    const LayerInfo* m_layer_info = nullptr;
    float m_qmin, m_qmax, m_fq;
    int m_nq = 0;
    bool m_is_barrel;

    // Data needed during setup
    struct HitInfo {
      float phi;
      float q;
    };
    std::vector<HitInfo> m_hit_infos;
    std::vector<uint32_t> m_qphifines;
  };

  //==============================================================================

  class EventOfHits {
  public:
    EventOfHits(const TrackerInfo& trk_inf);

    void reset() {
      for (auto& i : m_layers_of_hits) {
        i.reset();
      }
    }

    void suckInHits(int layer, const HitVec& hitv) { m_layers_of_hits[layer].suckInHits(hitv); }

    void suckInDeads(int layer, const DeadVec& deadv) { m_layers_of_hits[layer].suckInDeads(deadv); }

    const BeamSpot& refBeamSpot() const { return m_beam_spot; }
    void setBeamSpot(const BeamSpot& bs) { m_beam_spot = bs; }

    int nLayers() const { return m_n_layers; }

    LayerOfHits& operator[](int i) { return m_layers_of_hits[i]; }
    const LayerOfHits& operator[](int i) const { return m_layers_of_hits[i]; }

  private:
    std::vector<LayerOfHits> m_layers_of_hits;
    int m_n_layers;
    BeamSpot m_beam_spot;
  };

}  // end namespace mkfit
#endif
