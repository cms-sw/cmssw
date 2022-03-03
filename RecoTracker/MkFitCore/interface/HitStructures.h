#ifndef RecoTracker_MkFitCore_interface_HitStructures_h
#define RecoTracker_MkFitCore_interface_HitStructures_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
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

  inline bool sortHitsByPhiMT(const Hit& h1, const Hit& h2) {
    return std::atan2(h1.position()[1], h1.position()[0]) < std::atan2(h2.position()[1], h2.position()[0]);
  }

  inline bool sortTrksByPhiMT(const Track& t1, const Track& t2) { return t1.momPhi() < t2.momPhi(); }

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

    // Sub-detector type
    bool is_pixb_lyr() const { return m_layer_info->is_pixb_lyr(); }
    bool is_pixe_lyr() const { return m_layer_info->is_pixe_lyr(); }
    bool is_pix_lyr() const { return m_layer_info->is_pix_lyr(); }
    bool is_tib_lyr() const { return m_layer_info->is_tib_lyr(); }
    bool is_tob_lyr() const { return m_layer_info->is_tob_lyr(); }
    bool is_tid_lyr() const { return m_layer_info->is_tid_lyr(); }
    bool is_tec_lyr() const { return m_layer_info->is_tec_lyr(); }

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

  //==============================================================================
  // TrackCand, CombinedCandidate and EventOfCombinedCandidates
  //==============================================================================

  struct HoTNode {
    HitOnTrack m_hot;
    float m_chi2;
    int m_prev_idx;
  };

  struct HitMatch {
    int m_hit_idx = -1;
    int m_module_id = -1;
    float m_chi2 = 1e9;

    void reset() {
      m_hit_idx = -1;
      m_module_id = -1;
      m_chi2 = 1e9;
    }
  };

  struct HitMatchPair {
    HitMatch M[2];

    void reset() {
      M[0].reset();
      M[1].reset();
    }

    void consider_hit_for_overlap(int hit_idx, int module_id, float chi2) {
      if (module_id == M[0].m_module_id) {
        if (chi2 < M[0].m_chi2) {
          M[0].m_chi2 = chi2;
          M[0].m_hit_idx = hit_idx;
        }
      } else if (module_id == M[1].m_module_id) {
        if (chi2 < M[1].m_chi2) {
          M[1].m_chi2 = chi2;
          M[1].m_hit_idx = hit_idx;
        }
      } else {
        if (M[0].m_chi2 > M[1].m_chi2) {
          if (chi2 < M[0].m_chi2) {
            M[0] = {hit_idx, module_id, chi2};
          }
        } else {
          if (chi2 < M[1].m_chi2) {
            M[1] = {hit_idx, module_id, chi2};
          }
        }
      }
    }

    HitMatch* find_overlap(int hit_idx, int module_id) {
      if (module_id == M[0].m_module_id) {
        if (M[1].m_hit_idx >= 0)
          return &M[1];
      } else if (module_id == M[1].m_module_id) {
        if (M[0].m_hit_idx >= 0)
          return &M[0];
      } else {
        if (M[0].m_chi2 <= M[1].m_chi2) {
          if (M[0].m_hit_idx >= 0)
            return &M[0];
        } else {
          if (M[1].m_hit_idx >= 0)
            return &M[1];
        }
      }

      return nullptr;
    }
  };

  // CcPool - CombCandidate Pool and Allocator

  template <class T>
  class CcPool {
  public:
    void reset(std::size_t size) {
      if (size > m_mem.size())
        m_mem.resize(size);
      m_pos = 0;
      m_size = size;
    }

    void release() {
      std::vector<T> tmp;
      m_mem.swap(tmp);
      m_pos = 0;
      m_size = 0;
    }

    CcPool(std::size_t size = 0) {
      if (size)
        reset(size);
    }

    T* allocate(std::size_t n) {
      if (m_pos + n > m_size)
        throw std::bad_alloc();
      T* ret = &m_mem[m_pos];
      m_pos += n;
      return ret;
    }

    void deallocate(T* p, std::size_t n) noexcept {
      // we do not care, implied deallocation of the whole pool on reset().
    }

  private:
    std::vector<T> m_mem;
    std::size_t m_pos = 0;
    std::size_t m_size = 0;
  };

  template <class T>
  class CcAlloc {
  public:
    typedef T value_type;

    CcAlloc(CcPool<T>* p) : m_pool(p) {}

    const void* pool_id() const { return m_pool; }

    T* allocate(std::size_t n) { return m_pool->allocate(n); }

    void deallocate(T* p, std::size_t n) noexcept { m_pool->deallocate(p, n); }

  private:
    CcPool<T>* m_pool;
  };

  template <class T, class U>
  bool operator==(const CcAlloc<T>& a, const CcAlloc<U>& b) {
    return a.pool_id() == b.pool_id();
  }

  //------------------------------------------------------------------------------

  class CombCandidate;

  class TrackCand : public TrackBase {
  public:
    TrackCand() = default;

    explicit TrackCand(const TrackBase& base, CombCandidate* ccand) : TrackBase(base), m_comb_candidate(ccand) {
      // Reset hit counters -- caller has to initialize hits.
      lastHitIdx_ = -1;
      nFoundHits_ = 0;
    }

    // CombCandidate is used as a hit-container for a set of TrackCands originating from
    // the same seed and track building functions need this access to be able to add hits
    // into this holder class.
    // Access is guaranteed to be thread safe as seed ranges pointing into CombCandidate
    // vector is assigned to threads doing track-finding and final processing is only done
    // when all worker threads have finished.
    CombCandidate* combCandidate() const { return m_comb_candidate; }
    void setCombCandidate(CombCandidate* cc) { m_comb_candidate = cc; }

    int lastCcIndex() const { return lastHitIdx_; }
    int nFoundHits() const { return nFoundHits_; }
    int nMissingHits() const { return nMissingHits_; }
    int nOverlapHits() const { return nOverlapHits_; }
    int nTotalHits() const { return nFoundHits_ + nMissingHits_; }

    void setLastCcIndex(int i) { lastHitIdx_ = i; }
    void setNFoundHits(int n) { nFoundHits_ = n; }
    void setNMissingHits(int n) { nMissingHits_ = n; }
    void setNOverlapHits(int n) { nOverlapHits_ = n; }

    int nInsideMinusOneHits() const { return nInsideMinusOneHits_; }
    int nTailMinusOneHits() const { return nTailMinusOneHits_; }

    void setNInsideMinusOneHits(int n) { nInsideMinusOneHits_ = n; }
    void setNTailMinusOneHits(int n) { nTailMinusOneHits_ = n; }

    int originIndex() const { return m_origin_index; }
    void setOriginIndex(int oi) { m_origin_index = oi; }

    void resetOverlaps() { m_overlap_hits.reset(); }
    void considerHitForOverlap(int hit_idx, int module_id, float chi2) {
      m_overlap_hits.consider_hit_for_overlap(hit_idx, module_id, chi2);
    }
    HitMatch* findOverlap(int hit_idx, int module_id) { return m_overlap_hits.find_overlap(hit_idx, module_id); }

    // Inlines after definition of CombCandidate

    HitOnTrack getLastHitOnTrack() const;
    int getLastHitIdx() const;
    int getLastHitLyr() const;

    // For additional filter
    int getLastFoundPixelHitLyr() const;
    int getLastFoundHitLyr() const;
    int nUniqueLayers() const;

    int nLayersByTypeEncoded(const TrackerInfo& trk_inf) const;
    int nHitsByTypeEncoded(const TrackerInfo& trk_inf) const;

    int nPixelDecoded(const int& encoded) const { return encoded % 100; }
    int nStereoDecoded(const int& encoded) const { return (encoded / 100) % 100; }
    int nMonoDecoded(const int& encoded) const { return (encoded / 10000) % 100; }
    int nMatchedDecoded(const int& encoded) const { return encoded / 1000000; }
    int nTotMatchDecoded(const int& encoded) const {
      return encoded % 100 + (encoded / 100) % 100 + (encoded / 10000) % 100 - encoded / 1000000;
    }

    void addHitIdx(int hitIdx, int hitLyr, float chi2);

    HoTNode& refLastHoTNode();              // for filling up overlap info
    const HoTNode& refLastHoTNode() const;  // for dump traversal

    void incOverlapCount() { ++nOverlapHits_; }

    Track exportTrack(bool remove_missing_hits = false) const;

    void resetShortTrack() {
      score_ = getScoreWorstPossible();
      m_comb_candidate = nullptr;
    }

  private:
    CombCandidate* m_comb_candidate = nullptr;
    HitMatchPair m_overlap_hits;

    // using TrackBase::lastHitIdx_ to point into hit-on-track-node vector of CombCandidate
    short int nMissingHits_ = 0;
    short int nOverlapHits_ = 0;

    short int nInsideMinusOneHits_ = 0;
    short int nTailMinusOneHits_ = 0;

    short int m_origin_index = -1;  // index of origin candidate (used for overlaps in Standard)
  };

  inline bool sortByScoreTrackCand(const TrackCand& cand1, const TrackCand& cand2) {
    return cand1.score() > cand2.score();
  }

  inline float getScoreCand(const TrackCand& cand1, bool penalizeTailMissHits = false, bool inFindCandidates = false) {
    int nfoundhits = cand1.nFoundHits();
    int noverlaphits = cand1.nOverlapHits();
    int nmisshits = cand1.nInsideMinusOneHits();
    int ntailmisshits = penalizeTailMissHits ? cand1.nTailMinusOneHits() : 0;
    float pt = cand1.pT();
    float chi2 = cand1.chi2();
    // Do not allow for chi2<0 in score calculation
    if (chi2 < 0)
      chi2 = 0.f;
    return getScoreCalc(nfoundhits, ntailmisshits, noverlaphits, nmisshits, chi2, pt, inFindCandidates);
  }

  // CombCandidate -- a set of candidates from a given seed.

  class CombCandidate {
  public:
    using trk_cand_vec_type = std::vector<TrackCand, CcAlloc<TrackCand>>;
    using allocator_type = CcAlloc<TrackCand>;

    enum SeedState_e { Dormant = 0, Finding, Finished };

    CombCandidate(const allocator_type& alloc) : m_trk_cands(alloc), m_state(Dormant), m_pickup_layer(-1) {}

    // Required by std::uninitialized_fill_n when declaring vector<CombCandidate> in EventOfCombCandidates
    CombCandidate(const CombCandidate& o)
        : m_trk_cands(o.m_trk_cands),
          m_state(o.m_state),
          m_pickup_layer(o.m_pickup_layer),
          m_lastHitIdx_before_bkwsearch(o.m_lastHitIdx_before_bkwsearch),
          m_nInsideMinusOneHits_before_bkwsearch(o.m_nInsideMinusOneHits_before_bkwsearch),
          m_nTailMinusOneHits_before_bkwsearch(o.m_nTailMinusOneHits_before_bkwsearch),
#ifdef DUMPHITWINDOW
          m_seed_algo(o.m_seed_algo),
          m_seed_label(o.m_seed_label),
#endif
          m_hots_size(o.m_hots_size),
          m_hots(o.m_hots) {
    }

    // Required for std::swap().
    CombCandidate(CombCandidate&& o)
        : m_trk_cands(std::move(o.m_trk_cands)),
          m_best_short_cand(std::move(o.m_best_short_cand)),
          m_state(o.m_state),
          m_pickup_layer(o.m_pickup_layer),
          m_lastHitIdx_before_bkwsearch(o.m_lastHitIdx_before_bkwsearch),
          m_nInsideMinusOneHits_before_bkwsearch(o.m_nInsideMinusOneHits_before_bkwsearch),
          m_nTailMinusOneHits_before_bkwsearch(o.m_nTailMinusOneHits_before_bkwsearch),
#ifdef DUMPHITWINDOW
          m_seed_algo(o.m_seed_algo),
          m_seed_label(o.m_seed_label),
#endif
          m_hots_size(o.m_hots_size),
          m_hots(std::move(o.m_hots)) {
      // This is not needed as we do EOCC::reset() after EOCCS::resize which
      // calls Reset here and all CombCands get cleared.
      // However, if at some point we start using this for other purposes this needs
      // to be called as well.
      // for (auto &tc : *this) tc.setCombCandidate(this);
    }

    // Required for std::swap when filtering EventOfCombinedCandidates::m_candidates.
    // We do not call clear() on vectors as this will be done via EoCCs reset.
    // Probably would be better (clearer) if there was a special function that does
    // the swap in here or in EoCCs.
    CombCandidate& operator=(CombCandidate&& o) {
      m_trk_cands = (std::move(o.m_trk_cands));
      m_best_short_cand = std::move(o.m_best_short_cand);
      m_state = o.m_state;
      m_pickup_layer = o.m_pickup_layer;
      m_lastHitIdx_before_bkwsearch = o.m_lastHitIdx_before_bkwsearch;
      m_nInsideMinusOneHits_before_bkwsearch = o.m_nInsideMinusOneHits_before_bkwsearch;
      m_nTailMinusOneHits_before_bkwsearch = o.m_nTailMinusOneHits_before_bkwsearch;
#ifdef DUMPHITWINDOW
      m_seed_algo = o.m_seed_algo;
      m_seed_label = o.m_seed_label;
#endif
      m_hots_size = o.m_hots_size;
      m_hots = std::move(o.m_hots);

      for (auto& tc : m_trk_cands)
        tc.setCombCandidate(this);

      return *this;
    }

    // std::vector-like interface to access m_trk_cands
    bool empty() const { return m_trk_cands.empty(); }
    trk_cand_vec_type::size_type size() const { return m_trk_cands.size(); }
    void resize(trk_cand_vec_type::size_type count) { m_trk_cands.resize(count); }
    TrackCand& operator[](int i) { return m_trk_cands[i]; }
    const TrackCand& operator[](int i) const { return m_trk_cands[i]; }
    TrackCand& front() { return m_trk_cands.front(); }
    const TrackCand& front() const { return m_trk_cands.front(); }
    trk_cand_vec_type::reference emplace_back(TrackCand& tc) { return m_trk_cands.emplace_back(tc); }
    void clear() { m_trk_cands.clear(); }

    void reset(int max_cands_per_seed, int expected_num_hots) {
      std::vector<TrackCand, CcAlloc<TrackCand>> tmp(m_trk_cands.get_allocator());
      m_trk_cands.swap(tmp);
      m_trk_cands.reserve(max_cands_per_seed);  // we *must* never exceed this

      m_best_short_cand.setScore(getScoreWorstPossible());

      // state and pickup_layer set in importSeed.

      // expected_num_hots is different for CloneEngine and Std, especially as long as we
      // instantiate all candidates before purging them.
      // ce:  N_layer * N_cands ~~ 20 * 6 = 120
      // std: i don't know, maybe double?
      m_hots.reserve(expected_num_hots);
      m_hots_size = 0;
      m_hots.clear();
    }

    void importSeed(const Track& seed, int region);

    int addHit(const HitOnTrack& hot, float chi2, int prev_idx) {
      m_hots.push_back({hot, chi2, prev_idx});
      return m_hots_size++;
    }

    void mergeCandsAndBestShortOne(const IterationParams& params, bool update_score, bool sort_cands);

    void compactifyHitStorageForBestCand(bool remove_seed_hits, int backward_fit_min_hits);
    void beginBkwSearch();
    void endBkwSearch();

    // Accessors
    //-----------
    int hotsSize() const { return m_hots_size; }
    const HoTNode& hot_node(int i) const { return m_hots[i]; }
    HoTNode& hot_node_nc(int i) { return m_hots[i]; }
    HitOnTrack hot(int i) const { return m_hots[i].m_hot; }
    // Direct access into array for vectorized code in MkFinder
    const HoTNode* hotsData() const { return m_hots.data(); }

    const TrackCand& refBestShortCand() const { return m_best_short_cand; }
    void setBestShortCand(const TrackCand& tc) { m_best_short_cand = tc; }

    SeedState_e state() const { return m_state; }
    void setState(SeedState_e ss) { m_state = ss; }

    int pickupLayer() const { return m_pickup_layer; }

#ifdef DUMPHITWINDOW
    int seed_algo() const { return m_seed_algo; }
    int seed_label() const { return m_seed_label; }
#endif

  private:
    trk_cand_vec_type m_trk_cands;
    TrackCand m_best_short_cand;
    SeedState_e m_state : 8;
    int m_pickup_layer : 16;
    short int m_lastHitIdx_before_bkwsearch = -1;
    short int m_nInsideMinusOneHits_before_bkwsearch = -1;
    short int m_nTailMinusOneHits_before_bkwsearch = -1;

#ifdef DUMPHITWINDOW
    int m_seed_algo = 0;
    int m_seed_label = 0;
#endif
    int m_hots_size = 0;
    std::vector<HoTNode> m_hots;
  };

  //==============================================================================

  inline HitOnTrack TrackCand::getLastHitOnTrack() const { return m_comb_candidate->hot(lastHitIdx_); }

  inline int TrackCand::getLastHitIdx() const { return m_comb_candidate->hot(lastHitIdx_).index; }

  inline int TrackCand::getLastHitLyr() const { return m_comb_candidate->hot(lastHitIdx_).layer; }

  inline int TrackCand::getLastFoundHitLyr() const {
    int nh = nTotalHits();
    int ch = lastHitIdx_;
    int ll = -1;
    while (--nh >= 0) {
      const HoTNode& hot_node = m_comb_candidate->hot_node(ch);
      if (hot_node.m_hot.index < 0) {
        ch = hot_node.m_prev_idx;
      } else {
        ll = hot_node.m_hot.layer;
        break;
      }
    }
    return ll;
  }

  inline int TrackCand::getLastFoundPixelHitLyr() const {
    int nh = nTotalHits();
    int ch = lastHitIdx_;
    int ll = -1;
    while (--nh >= 0) {
      const HoTNode& hot_node = m_comb_candidate->hot_node(ch);
      int tl = hot_node.m_hot.layer;
      if (hot_node.m_hot.index < 0 || !((0 <= tl && tl <= 3) || (18 <= tl && tl <= 20) || (45 <= tl && tl <= 47))) {
        ch = hot_node.m_prev_idx;
      } else if ((0 <= tl && tl <= 3) || (18 <= tl && tl <= 20) || (45 <= tl && tl <= 47)) {
        ll = hot_node.m_hot.layer;
        break;
      }
    }
    return ll;
  }

  inline int TrackCand::nUniqueLayers() const {
    int nUL = 0;
    int prevL = -1;
    int nh = nTotalHits();
    int ch = lastHitIdx_;

    while (--nh >= 0) {
      const HoTNode& hot_node = m_comb_candidate->hot_node(ch);
      int thisL = hot_node.m_hot.layer;
      if (thisL >= 0 && (hot_node.m_hot.index >= 0 || hot_node.m_hot.index == -9) && thisL != prevL) {
        ++nUL;
        prevL = thisL;
      }
      ch = hot_node.m_prev_idx;
    }
    return nUL;
  }

  inline int TrackCand::nHitsByTypeEncoded(const TrackerInfo& trk_inf) const {
    int prevL = -1;
    bool prevStereo = false;
    int nh = nTotalHits();
    int ch = lastHitIdx_;
    int pix = 0, stereo = 0, mono = 0, matched = 0;
    int doubleStereo = -1;
    while (--nh >= 0) {
      const HoTNode& hot_node = m_comb_candidate->hot_node(ch);
      int thisL = hot_node.m_hot.layer;
      if (thisL >= 0 && (hot_node.m_hot.index >= 0 || hot_node.m_hot.index == -9)) {
        if (trk_inf.is_pix_lyr(thisL))
          ++pix;
        else if (trk_inf.is_stereo(thisL)) {
          ++stereo;
          if (thisL == prevL)
            doubleStereo = thisL;
        } else {
          //mono if not pixel, nor stereo - can be matched to stereo
          ++mono;
          if (prevStereo && thisL == prevL - 1)
            ++matched;
          else if (thisL == prevL && thisL == doubleStereo - 1)
            ++matched;  //doubleMatch, the first is counted early on
        }
        prevL = thisL;
        prevStereo = stereo;
      }
      ch = hot_node.m_prev_idx;
    }
    return pix + 100 * stereo + 10000 * mono + 1000000 * matched;
  }

  inline int TrackCand::nLayersByTypeEncoded(const TrackerInfo& trk_inf) const {
    int prevL = -1;
    bool prevStereo = false;
    int nh = nTotalHits();
    int ch = lastHitIdx_;
    int pix = 0, stereo = 0, mono = 0, matched = 0;
    while (--nh >= 0) {
      const HoTNode& hot_node = m_comb_candidate->hot_node(ch);
      int thisL = hot_node.m_hot.layer;
      if (thisL >= 0 && (hot_node.m_hot.index >= 0 || hot_node.m_hot.index == -9) && thisL != prevL) {
        if (trk_inf.is_pix_lyr(thisL))
          ++pix;
        else if (trk_inf.is_stereo(thisL))
          ++stereo;
        else {
          //mono if not pixel, nor stereo - can be matched to stereo
          ++mono;
          if (prevStereo && thisL == prevL - 1)
            ++matched;
        }
        prevL = thisL;
        prevStereo = stereo;
      }
      ch = hot_node.m_prev_idx;
    }
    return pix + 100 * stereo + 10000 * mono + 1000000 * matched;
  }

  inline HoTNode& TrackCand::refLastHoTNode() { return m_comb_candidate->hot_node_nc(lastHitIdx_); }

  inline const HoTNode& TrackCand::refLastHoTNode() const { return m_comb_candidate->hot_node(lastHitIdx_); }

  //------------------------------------------------------------------------------

  inline void TrackCand::addHitIdx(int hitIdx, int hitLyr, float chi2) {
    lastHitIdx_ = m_comb_candidate->addHit({hitIdx, hitLyr}, chi2, lastHitIdx_);

    if (hitIdx >= 0 || hitIdx == -9) {
      ++nFoundHits_;
      chi2_ += chi2;
      nInsideMinusOneHits_ += nTailMinusOneHits_;
      nTailMinusOneHits_ = 0;
    }
    //Note that for tracks passing through an inactive module (hitIdx = -7), we do not count the -7 hit against the track when scoring.
    else {
      ++nMissingHits_;
      if (hitIdx == -1)
        ++nTailMinusOneHits_;
    }
  }

  //==============================================================================

  class EventOfCombCandidates {
  public:
    EventOfCombCandidates(int size = 0) : m_cc_pool(), m_candidates(), m_capacity(0), m_size(0) {}

    void releaseMemory() {
      {  // Get all the destructors called before nuking CcPool.
        std::vector<CombCandidate> tmp;
        m_candidates.swap(tmp);
      }
      m_capacity = 0;
      m_size = 0;
      m_cc_pool.release();
    }

    void reset(int new_capacity, int max_cands_per_seed, int expected_num_hots = 128) {
      m_cc_pool.reset(new_capacity * max_cands_per_seed);
      if (new_capacity > m_capacity) {
        CcAlloc<TrackCand> alloc(&m_cc_pool);
        std::vector<CombCandidate> tmp(new_capacity, alloc);
        m_candidates.swap(tmp);
        m_capacity = new_capacity;
      }
      for (int s = 0; s < new_capacity; ++s) {
        m_candidates[s].reset(max_cands_per_seed, expected_num_hots);
      }
      for (int s = new_capacity; s < m_capacity; ++s) {
        m_candidates[s].reset(0, 0);
      }

      m_size = 0;
    }

    void resizeAfterFiltering(int n_removed) {
      assert(n_removed <= m_size);
      m_size -= n_removed;
    }

    void insertSeed(const Track& seed, int region) {
      assert(m_size < m_capacity);

      m_candidates[m_size].importSeed(seed, region);

      ++m_size;
    }

    void compactifyHitStorageForBestCand(bool remove_seed_hits, int backward_fit_min_hits) {
      for (int i = 0; i < m_size; ++i)
        m_candidates[i].compactifyHitStorageForBestCand(remove_seed_hits, backward_fit_min_hits);
    }

    void beginBkwSearch() {
      for (int i = 0; i < m_size; ++i)
        m_candidates[i].beginBkwSearch();
    }
    void endBkwSearch() {
      for (int i = 0; i < m_size; ++i)
        m_candidates[i].endBkwSearch();
    }

    // Accessors
    int size() const { return m_size; }

    const CombCandidate& operator[](int i) const { return m_candidates[i]; }
    CombCandidate& operator[](int i) { return m_candidates[i]; }
    CombCandidate& cand(int i) { return m_candidates[i]; }

    // Direct access for vectorized functions in MkBuilder / MkFinder
    const std::vector<CombCandidate>& refCandidates() const { return m_candidates; }
    std::vector<CombCandidate>& refCandidates_nc() { return m_candidates; }

  private:
    CcPool<TrackCand> m_cc_pool;

    std::vector<CombCandidate> m_candidates;

    int m_capacity;
    int m_size;
  };

}  // end namespace mkfit
#endif
