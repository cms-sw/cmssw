#ifndef RecoTracker_MkFitCore_interface_IterationConfig_h
#define RecoTracker_MkFitCore_interface_IterationConfig_h

#include "RecoTracker/MkFitCore/interface/FunctionTypes.h"
#include "RecoTracker/MkFitCore/interface/SteeringParams.h"

#include "nlohmann/json_fwd.hpp"

#include <functional>

namespace mkfit {

  //==============================================================================
  // Hit masks / IterationMaskIfc
  //==============================================================================

  struct IterationMaskIfcBase {
    virtual ~IterationMaskIfcBase() {}

    virtual const std::vector<bool> *get_mask_for_layer(int layer) const { return nullptr; }
  };

  struct IterationMaskIfc : public IterationMaskIfcBase {
    std::vector<std::vector<bool>> m_mask_vector;

    const std::vector<bool> *get_mask_for_layer(int layer) const override { return &m_mask_vector[layer]; }
  };

  //==============================================================================
  // IterationLayerConfig
  //==============================================================================

  class IterationConfig;

  class IterationLayerConfig {
  public:
    int m_layer = -1;
    // Selection limits.
    float m_select_min_dphi;
    float m_select_max_dphi;
    float m_select_min_dq;
    float m_select_max_dq;

    void set_selection_limits(float p1, float p2, float q1, float q2) {
      m_select_min_dphi = p1;
      m_select_max_dphi = p2;
      m_select_min_dq = q1;
      m_select_max_dq = q2;
    }

    //----------------------------------------------------------------------------

    float min_dphi() const { return m_select_min_dphi; }
    float max_dphi() const { return m_select_max_dphi; }
    float min_dq() const { return m_select_min_dq; }
    float max_dq() const { return m_select_max_dq; }

    const std::vector<float> &get_window_params(bool forward, bool fallback_to_other) const {
      if (fallback_to_other) {
        // Empty vector is a valid result, we do not need to check both.
        if (forward)
          return m_winpars_fwd.empty() ? m_winpars_bkw : m_winpars_fwd;
        else
          return m_winpars_bkw.empty() ? m_winpars_fwd : m_winpars_bkw;
      } else {
        return forward ? m_winpars_fwd : m_winpars_bkw;
      }
    }

    //Hit selection window parameters: 2D fit/layer (72 in phase-1 CMS geometry).
    //Used in MkFinder::selectHitIndices().
    std::vector<float> m_winpars_fwd;
    std::vector<float> m_winpars_bkw;

    //----------------------------------------------------------------------------

    IterationLayerConfig() {}
  };

  //==============================================================================
  // IterationParams
  //==============================================================================

  class IterationParams {
  public:
    int nlayers_per_seed = 3;
    int maxCandsPerSeed = 5;
    int maxHolesPerCand = 4;
    int maxConsecHoles = 1;
    float chi2Cut_min = 15.0;
    float chi2CutOverlap = 3.5;
    float pTCutOverlap = 0.0;

    //quality filter params
    int minHitsQF = 4;

    //min pT cut
    float minPtCut = 0.0;

    //max cluster size cut for SiStrip hits
    unsigned int maxClusterSize = 8;
  };

  //==============================================================================
  // IterationSeedPartition
  //==============================================================================

  class IterationSeedPartition {
  public:
    using register_seed_phi_eta_foo = void(float, float);

    std::vector<int> m_region;
    std::function<register_seed_phi_eta_foo> m_phi_eta_foo;

    IterationSeedPartition(int size) : m_region(size) {}
  };

  //==============================================================================
  // IterationConfig
  //==============================================================================

  class IterationConfig {
  public:
    int m_iteration_index = -1;
    int m_track_algorithm = -1;

    bool m_requires_seed_hit_sorting = false;

    bool m_backward_search = false;
    bool m_backward_drop_seed_hits = false;

    int m_backward_fit_min_hits = -1;  // Min number of hits to keep when m_backward_drop_seed_hits is true

    // seed cleaning params with good defaults (all configurable)
    float sc_ptthr_hpt = 2.0;
    float sc_drmax_bh = 0.010;
    float sc_dzmax_bh = 0.005;
    float sc_drmax_eh = 0.020;
    float sc_dzmax_eh = 0.020;
    float sc_drmax_bl = 0.010;
    float sc_dzmax_bl = 0.005;
    float sc_drmax_el = 0.030;
    float sc_dzmax_el = 0.030;

    // duplicate cleaning params with good defaults (all configurable)
    float dc_fracSharedHits = 0.19;
    float dc_drth_central = 0.001;
    float dc_drth_obarrel = 0.001;
    float dc_drth_forward = 0.001;

    // Iteration parameters (could be a ptr)
    IterationParams m_params;
    IterationParams m_backward_params;

    int m_n_regions = -1;
    std::vector<int> m_region_order;
    std::vector<SteeringParams> m_steering_params;
    std::vector<IterationLayerConfig> m_layer_configs;

    // *** Standard functions
    // - seed cleaning: called directly from top-level per-iteration steering code.
    clean_seeds_func m_seed_cleaner;
    // - seed partitioning into eta regions: called from MkBuilder::find_tracks_load_seeds().
    partition_seeds_func m_seed_partitioner;
    // - candidate filtering: passed to MkBuilder::filter_comb_cands().
    filter_candidates_func m_pre_bkfit_filter, m_post_bkfit_filter;
    // - duplicate cleaning: called directly from top-level per-iteration steering code.
    clean_duplicates_func m_duplicate_cleaner;
    // - default track scoring function, can be overriden in SteeringParams for each eta region.
    track_score_func m_default_track_scorer;

    // Names for Standard functions that get saved to / loaded from JSON.
    std::string m_seed_cleaner_name;
    std::string m_seed_partitioner_name;
    std::string m_pre_bkfit_filter_name, m_post_bkfit_filter_name;
    std::string m_duplicate_cleaner_name;
    std::string m_default_track_scorer_name = "default";

    //----------------------------------------------------------------------------

    IterationConfig() {}

    // -------- Getter functions

    IterationLayerConfig &layer(int i) { return m_layer_configs[i]; }
    SteeringParams &steering_params(int region) { return m_steering_params[region]; }

    bool merge_seed_hits_during_cleaning() const { return m_backward_search && m_backward_drop_seed_hits; }

    // -------- Setup functions

    void setupStandardFunctionsFromNames();

    void cloneLayerSteerCore(const IterationConfig &o) {
      // Clone common settings for an iteration.
      // m_iteration_index, m_track_algorithm, cleaning and bkw-search flags,
      // and IterationParams are not copied.
      // Standard functions are also not copied, only their names so one should
      // call setupStandardFunctionsFromNames() later on.

      m_n_regions = o.m_n_regions;
      m_region_order = o.m_region_order;
      m_steering_params = o.m_steering_params;
      m_layer_configs = o.m_layer_configs;

      m_seed_cleaner_name = o.m_seed_cleaner_name;
      m_seed_partitioner_name = o.m_seed_partitioner_name;
      m_pre_bkfit_filter_name = o.m_pre_bkfit_filter_name;
      m_post_bkfit_filter_name = o.m_post_bkfit_filter_name;
      m_duplicate_cleaner_name = o.m_duplicate_cleaner_name;
      m_default_track_scorer_name = o.m_default_track_scorer_name;
    }

    void set_iteration_index_and_track_algorithm(int idx, int trk_alg) {
      m_iteration_index = idx;
      m_track_algorithm = trk_alg;
    }

    void set_dupl_params(float sharedFrac, float drthCentral, float drthObarrel, float drthForward) {
      dc_fracSharedHits = sharedFrac;
      dc_drth_central = drthCentral;
      dc_drth_obarrel = drthObarrel;
      dc_drth_forward = drthForward;
    }

    void set_seed_cleaning_params(float pt_thr,
                                  float dzmax_bh,
                                  float drmax_bh,
                                  float dzmax_bl,
                                  float drmax_bl,
                                  float dzmax_eh,
                                  float drmax_eh,
                                  float dzmax_el,
                                  float drmax_el) {
      sc_ptthr_hpt = pt_thr;
      sc_drmax_bh = drmax_bh;
      sc_dzmax_bh = dzmax_bh;
      sc_drmax_eh = drmax_eh;
      sc_dzmax_eh = dzmax_eh;
      sc_drmax_bl = drmax_bl;
      sc_dzmax_bl = dzmax_bl;
      sc_drmax_el = drmax_el;
      sc_dzmax_el = dzmax_el;
    }

    void set_num_regions_layers(int nreg, int nlay) {
      m_n_regions = nreg;
      m_region_order.resize(nreg);
      m_steering_params.resize(nreg);
      for (int i = 0; i < nreg; ++i)
        m_steering_params[i].m_region = i;
      m_layer_configs.resize(nlay);
      for (int i = 0; i < nlay; ++i)
        m_layer_configs[i].m_layer = i;
    }

    // Catalog of Standard functions
    static void register_seed_cleaner(const std::string &name, clean_seeds_func func);
    static void register_seed_partitioner(const std::string &name, partition_seeds_func func);
    static void register_candidate_filter(const std::string &name, filter_candidates_func func);
    static void register_duplicate_cleaner(const std::string &name, clean_duplicates_func func);
    static void register_track_scorer(const std::string &name, track_score_func func);

    static clean_seeds_func get_seed_cleaner(const std::string &name);
    static partition_seeds_func get_seed_partitioner(const std::string &name);
    static filter_candidates_func get_candidate_filter(const std::string &name);
    static clean_duplicates_func get_duplicate_cleaner(const std::string &name);
    static track_score_func get_track_scorer(const std::string &name);
  };

  //==============================================================================
  // IterationsInfo
  //==============================================================================

  class IterationsInfo {
  public:
    std::vector<IterationConfig> m_iterations;

    IterationsInfo() {}

    void resize(int ni) { m_iterations.resize(ni); }

    int size() const { return m_iterations.size(); }

    IterationConfig &operator[](int i) { return m_iterations[i]; }
    const IterationConfig &operator[](int i) const { return m_iterations[i]; }

    void setupStandardFunctionsFromNames() {
      for (auto &i : m_iterations)
        i.setupStandardFunctionsFromNames();
    }
  };

  //==============================================================================

  // IterationConfig instances are created in Geoms/CMS-phase1.cc, Create_CMS_phase1(),
  // filling the IterationsInfo object passed in by reference.

  //==============================================================================
  // JSON config interface
  //==============================================================================

  class ConfigJsonPatcher {
  public:
    struct PatchReport {
      int n_files = 0;
      int n_json_entities = 0;
      int n_replacements = 0;

      void inc_counts(int f, int e, int r) {
        n_files += f;
        n_json_entities += e;
        n_replacements += r;
      }
      void inc_counts(const PatchReport &pr) {
        n_files += pr.n_files;
        n_json_entities += pr.n_json_entities;
        n_replacements += pr.n_replacements;
      }
      void reset() { n_files = n_json_entities = n_replacements = 0; }
    };

  private:
    std::unique_ptr<nlohmann::json> m_json;
    nlohmann::json *m_current = nullptr;

    // add stack and cd_up() ? also, name stack for exceptions and printouts
    std::vector<nlohmann::json *> m_json_stack;
    std::vector<std::string> m_path_stack;

    bool m_verbose = false;

    std::string get_abs_path() const;
    std::string exc_hdr(const char *func = nullptr) const;

  public:
    ConfigJsonPatcher(bool verbose = false);
    ~ConfigJsonPatcher();

    template <class T>
    void load(const T &o);
    template <class T>
    void save(T &o);

    void cd(const std::string &path);
    void cd_up(const std::string &path = "");
    void cd_top(const std::string &path = "");

    template <typename T>
    void replace(const std::string &path, T val);

    template <typename T>
    void replace(int first, int last, const std::string &path, T val);

    nlohmann::json &get(const std::string &path);

    int replace(const nlohmann::json &j);

    std::string dump(int indent = 2);
  };

  class ConfigJson {
  public:
    ConfigJson(bool verbose = false) : m_verbose(verbose) {}

    // Patch IterationsInfo from a vector of files.
    // Assumes patch files include iteration-info preambles, i.e., they
    // were saved with include_iter_info_preamble=true.
    // If report is non-null counts are added to existing object.
    void patch_Files(IterationsInfo &its_info,
                     const std::vector<std::string> &fnames,
                     ConfigJsonPatcher::PatchReport *report = nullptr);

    // Load a single iteration from JSON file.
    // Searches for a match between m_algorithm in its_info and in JSON file to decide
    // which IterationConfig it will clone and patch-load the JSON file over.
    // The IterationConfig in question *must* match in structure to what is on file,
    // in particular, arrays must be of same lengths.
    // Assumes JSON file has been saved WITHOUT iteration-info preamble.
    // Returns a unique_ptr to the cloned IterationConfig.
    // If report is non-null counts are added to existing object.
    std::unique_ptr<IterationConfig> patchLoad_File(const IterationsInfo &its_info,
                                                    const std::string &fname,
                                                    ConfigJsonPatcher::PatchReport *report = nullptr);

    // Load a single iteration from JSON file.
    // This leaves IterationConfig data-members that are not registered
    // in JSON schema at their default values.
    // There are several std::function members like this.
    // Assumes JSON file has been saved WITHOUT iteration-info preamble.
    // Returns a unique_ptr to the cloned IterationConfig.
    std::unique_ptr<IterationConfig> load_File(const std::string &fname);

    void save_Iterations(IterationsInfo &its_info, const std::string &fname_fmt, bool include_iter_info_preamble);

    void dump(IterationsInfo &its_info);

    void test_Direct(IterationConfig &it_cfg);
    void test_Patcher(IterationConfig &it_cfg);

  private:
    bool m_verbose = false;
  };
}  // end namespace mkfit

#endif
