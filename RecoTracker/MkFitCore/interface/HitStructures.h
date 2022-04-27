#ifndef RecoTracker_MkFitCore_interface_HitStructures_h
#define RecoTracker_MkFitCore_interface_HitStructures_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/binnor.h"

namespace mkfit {

  class IterationParams;

  //==============================================================================
  // LayerOfHits
  //==============================================================================

  // Note: the same code is used for barrel and endcap. In barrel the longitudinal
  // bins are in Z and in endcap they are in R -- here this coordinate is called Q.

  // When COPY_SORTED_HITS is not defined, hits are accessed from the original hit
  // vector and only sort ranks are kept for proper access.
  // #define COPY_SORTED_HITS

  class LayerOfHits {
  public:
    using bin_index_t = unsigned short;
    using bin_content_t = unsigned int;
    using axis_phi_t = axis_pow2_u1<float, bin_index_t, 16, 8>;
    using axis_eta_t = axis<float, bin_index_t, 16, 8>;
    using binnor_t = binnor<bin_content_t, axis_phi_t, axis_eta_t, 18, 14>;

    // Initializator

    struct Initializator {
      const LayerInfo& m_linfo;
      float m_qmin, m_qmax;
      unsigned int m_nq;

      void setup(float qmin, float qmax, float dq);

      Initializator(const LayerInfo& li, float qmin, float qmax, unsigned int nq);
      Initializator(const LayerInfo& li, float qmin, float qmax, float dq);
      Initializator(const LayerInfo& li);
    };

    // Constructor

    LayerOfHits(const LayerOfHits::Initializator& i);

    ~LayerOfHits();

    // Setup and filling
    //-------------------

    void reset() {}

    // Get in all hits from given hit-vec
    void suckInHits(const HitVec& hitv);

    // Get in all dead regions from given dead-vec
    void suckInDeads(const DeadVec& deadv);

    // Use external hit-vec and only use hits that are passed to me.
    void beginRegistrationOfHits(const HitVec& hitv);
    void registerHit(unsigned int idx);
    void endRegistrationOfHits(bool build_original_to_internal_map);

    unsigned int nHits() const { return m_n_hits; }

    // Bin access / queries
    //----------------------
    bin_index_t qBin(float q) const { return m_ax_eta.from_R_to_N_bin(q); }
    bin_index_t qBinChecked(float q) const { return m_ax_eta.from_R_to_N_bin_safe(q); }

    // if you don't pass phi in (-pi, +pi), mask away the upper bits using m_phi_mask or use the Checked version.
    bin_index_t phiBin(float phi) const { return m_ax_phi.from_R_to_N_bin(phi); }
    bin_index_t phiBinChecked(float phi) const { return m_ax_phi.from_R_to_N_bin_safe(phi); }

    bin_index_t phiMaskApply(bin_index_t in) const { return in & m_ax_phi.c_N_mask; }

    binnor_t::C_pair phiQBinContent(bin_index_t pi, bin_index_t qi) const { return m_binnor.get_content(pi, qi); }

    bool isBinDead(bin_index_t pi, bin_index_t qi) const { return m_dead_bins[qi * m_ax_phi.size_of_N() + pi]; }

    float hit_q(unsigned int i) const { return m_hit_qs[i]; }
    float hit_phi(unsigned int i) const { return m_hit_phis[i]; }

    // Use this to map original indices to sorted internal ones. m_ext_idcs needs to be initialized.
    unsigned int getHitIndexFromOriginal(unsigned int i) const { return m_ext_idcs[i - m_min_ext_idx]; }
    // Use this to remap internal hit index to external one.
    unsigned int getOriginalHitIndex(unsigned int i) const { return m_binnor.m_ranks[i]; }

#ifdef COPY_SORTED_HITS
    const Hit& refHit(int i) const { return m_hits[i]; }
    const Hit* hitArray() const { return m_hits; }
#else
    const Hit& refHit(int i) const { return (*m_ext_hits)[i]; }
    const Hit* hitArray() const { return m_ext_hits->data(); }
#endif

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
    axis_phi_t m_ax_phi;
    axis_eta_t m_ax_eta;
    binnor_t m_binnor;

#ifdef COPY_SORTED_HITS
    void alloc_hits(int size);
    void free_hits()

        Hit* m_hits = nullptr;
    int m_capacity = 0;
#else
    const HitVec* m_ext_hits;
#endif
    unsigned int* m_hit_ranks = nullptr;  // allocated by IceSort via new []
    std::vector<unsigned int> m_ext_idcs;
    unsigned int m_min_ext_idx, m_max_ext_idx;
    unsigned int m_n_hits = 0;

    // Bin information for dead regions
    std::vector<bool> m_dead_bins;

    // Cached hit phi and q values to minimize Hit memory access
    std::vector<float> m_hit_phis;
    std::vector<float> m_hit_qs;

    // Geometry / q-binning constants - initialized in setupLayer()
    const LayerInfo* m_layer_info = nullptr;
    bool m_is_barrel;

    // Data needed during setup
    struct HitInfo {
      float phi;
      float q;
    };
    std::vector<HitInfo> m_hit_infos;
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
