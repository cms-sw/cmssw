#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

//#define DEBUG
#include "Debug.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <mutex>
#include <regex>
#include <iostream>
#include <iomanip>

// Redefine to also support ordered_json ... we want to keep variable order in JSON save files.
#define ITCONF_DEFINE_TYPE_NON_INTRUSIVE(Type, ...)                                             \
  inline void to_json(nlohmann::json &nlohmann_json_j, const Type &nlohmann_json_t) {           \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__))                    \
  }                                                                                             \
  inline void from_json(const nlohmann::json &nlohmann_json_j, Type &nlohmann_json_t) {         \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__))                  \
  }                                                                                             \
  inline void to_json(nlohmann::ordered_json &nlohmann_json_j, const Type &nlohmann_json_t) {   \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__))                    \
  }                                                                                             \
  inline void from_json(const nlohmann::ordered_json &nlohmann_json_j, Type &nlohmann_json_t) { \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__))                  \
  }

namespace mkfit {

  // Begin AUTO code, some members commented out.

  ITCONF_DEFINE_TYPE_NON_INTRUSIVE(mkfit::LayerControl,
                                   /* int   */ m_layer)

  ITCONF_DEFINE_TYPE_NON_INTRUSIVE(mkfit::SteeringParams,
                                   /* std::vector<LayerControl> */ m_layer_plan,
                                   /* std::string */ m_track_scorer_name,
                                   /* int */ m_region,
                                   /* int */ m_fwd_search_pickup,
                                   /* int */ m_bkw_fit_last,
                                   /* int */ m_bkw_search_pickup

  )

  ITCONF_DEFINE_TYPE_NON_INTRUSIVE(mkfit::IterationLayerConfig,
                                   /* int */ m_layer,
                                   /* float */ m_select_min_dphi,
                                   /* float */ m_select_max_dphi,
                                   /* float */ m_select_min_dq,
                                   /* float */ m_select_max_dq,
                                   /* float */ c_dp_sf,
                                   /* float */ c_dp_0,
                                   /* float */ c_dp_1,
                                   /* float */ c_dp_2,
                                   /* float */ c_dq_sf,
                                   /* float */ c_dq_0,
                                   /* float */ c_dq_1,
                                   /* float */ c_dq_2,
                                   /* float */ c_c2_sf,
                                   /* float */ c_c2_0,
                                   /* float */ c_c2_1,
                                   /* float */ c_c2_2)

  ITCONF_DEFINE_TYPE_NON_INTRUSIVE(mkfit::IterationParams,
                                   /* int */ nlayers_per_seed,
                                   /* int */ maxCandsPerSeed,
                                   /* int */ maxHolesPerCand,
                                   /* int */ maxConsecHoles,
                                   /* float */ chi2Cut_min,
                                   /* float */ chi2CutOverlap,
                                   /* float */ pTCutOverlap,
                                   /* int */ minHitsQF,
                                   /* float */ minPtCut,
                                   /* unsigned int */ maxClusterSize)

  ITCONF_DEFINE_TYPE_NON_INTRUSIVE(mkfit::IterationConfig,
                                   /* int */ m_iteration_index,
                                   /* int */ m_track_algorithm,
                                   /* std::string */ m_seed_cleaner_name,
                                   /* std::string */ m_seed_partitioner_name,
                                   /* std::string */ m_pre_bkfit_filter_name,
                                   /* std::string */ m_post_bkfit_filter_name,
                                   /* std::string */ m_duplicate_cleaner_name,
                                   /* std::string */ m_default_track_scorer_name,
                                   /* bool */ m_requires_seed_hit_sorting,
                                   /* bool */ m_backward_search,
                                   /* bool */ m_backward_drop_seed_hits,
                                   /* int */ m_backward_fit_min_hits,
                                   /* float */ sc_ptthr_hpt,
                                   /* float */ sc_drmax_bh,
                                   /* float */ sc_dzmax_bh,
                                   /* float */ sc_drmax_eh,
                                   /* float */ sc_dzmax_eh,
                                   /* float */ sc_drmax_bl,
                                   /* float */ sc_dzmax_bl,
                                   /* float */ sc_drmax_el,
                                   /* float */ sc_dzmax_el,
                                   /* float */ dc_fracSharedHits,
                                   /* float */ dc_drth_central,
                                   /* float */ dc_drth_obarrel,
                                   /* float */ dc_drth_forward,
                                   /* mkfit::IterationParams */ m_params,
                                   /* mkfit::IterationParams */ m_backward_params,
                                   /* int */ m_n_regions,
                                   /* vector<int> */ m_region_order,
                                   /* vector<mkfit::SteeringParams> */ m_steering_params,
                                   /* vector<mkfit::IterationLayerConfig> */ m_layer_configs)

  ITCONF_DEFINE_TYPE_NON_INTRUSIVE(mkfit::IterationsInfo,
                                   /* vector<mkfit::IterationConfig> */ m_iterations)

  // End AUTO code.

  // Begin IterationConfig function catalogs

  namespace {
    struct FuncCatalog {
      std::map<std::string, clean_seeds_func> seed_cleaners;
      std::map<std::string, partition_seeds_func> seed_partitioners;
      std::map<std::string, filter_candidates_func> candidate_filters;
      std::map<std::string, clean_duplicates_func> duplicate_cleaners;
      std::map<std::string, track_score_func> track_scorers;

      std::mutex catalog_mutex;
    };

    FuncCatalog &get_catalog() {
      CMS_SA_ALLOW static FuncCatalog func_catalog;
      return func_catalog;
    }
  }  // namespace

#define GET_FC              \
  auto &fc = get_catalog(); \
  const std::lock_guard<std::mutex> lock(fc.catalog_mutex)

  void IterationConfig::register_seed_cleaner(const std::string &name, clean_seeds_func func) {
    GET_FC;
    fc.seed_cleaners.insert({name, func});
  }
  void IterationConfig::register_seed_partitioner(const std::string &name, partition_seeds_func func) {
    GET_FC;
    fc.seed_partitioners.insert({name, func});
  }
  void IterationConfig::register_candidate_filter(const std::string &name, filter_candidates_func func) {
    GET_FC;
    fc.candidate_filters.insert({name, func});
  }
  void IterationConfig::register_duplicate_cleaner(const std::string &name, clean_duplicates_func func) {
    GET_FC;
    fc.duplicate_cleaners.insert({name, func});
  }
  void IterationConfig::register_track_scorer(const std::string &name, track_score_func func) {
    GET_FC;
    fc.track_scorers.insert({name, func});
  }

  namespace {
    template <class T>
    typename T::mapped_type resolve_func_name(const T &cont, const std::string &name, const char *func) {
      if (name.empty()) {
        return nullptr;
      }
      auto ii = cont.find(name);
      if (ii == cont.end()) {
        std::string es(func);
        es += " '" + name + "' not found in function registry.";
        throw std::runtime_error(es);
      }
      return ii->second;
    }
  }  // namespace

  clean_seeds_func IterationConfig::get_seed_cleaner(const std::string &name) {
    GET_FC;
    return resolve_func_name(fc.seed_cleaners, name, __func__);
  }
  partition_seeds_func IterationConfig::get_seed_partitioner(const std::string &name) {
    GET_FC;
    return resolve_func_name(fc.seed_partitioners, name, __func__);
  }
  filter_candidates_func IterationConfig::get_candidate_filter(const std::string &name) {
    GET_FC;
    return resolve_func_name(fc.candidate_filters, name, __func__);
  }
  clean_duplicates_func IterationConfig::get_duplicate_cleaner(const std::string &name) {
    GET_FC;
    return resolve_func_name(fc.duplicate_cleaners, name, __func__);
  }
  track_score_func IterationConfig::get_track_scorer(const std::string &name) {
    GET_FC;
    return resolve_func_name(fc.track_scorers, name, __func__);
  }

#undef GET_FC

  // End IterationConfig function catalogs

  void IterationConfig::setupStandardFunctionsFromNames() {
    m_seed_cleaner = get_seed_cleaner(m_seed_cleaner_name);
    dprintf(" Set seed_cleaner for '%s' %s\n", m_seed_cleaner_name.c_str(), m_seed_cleaner ? "SET" : "NOT SET");

    m_seed_partitioner = get_seed_partitioner(m_seed_partitioner_name);
    dprintf(
        " Set seed_partitioner for '%s' %s\n", m_seed_partitioner_name.c_str(), m_seed_partitioner ? "SET" : "NOT SET");

    m_pre_bkfit_filter = get_candidate_filter(m_pre_bkfit_filter_name);
    dprintf(
        " Set pre_bkfit_filter for '%s' %s\n", m_pre_bkfit_filter_name.c_str(), m_pre_bkfit_filter ? "SET" : "NOT SET");

    m_post_bkfit_filter = get_candidate_filter(m_post_bkfit_filter_name);
    dprintf(" Set post_bkfit_filter for '%s' %s\n",
            m_post_bkfit_filter_name.c_str(),
            m_post_bkfit_filter ? "SET" : "NOT SET");

    m_duplicate_cleaner = get_duplicate_cleaner(m_duplicate_cleaner_name);
    dprintf(" Set duplicate_cleaner for '%s' %s\n",
            m_duplicate_cleaner_name.c_str(),
            m_duplicate_cleaner ? "SET" : "NOT SET");

    m_default_track_scorer = get_track_scorer(m_default_track_scorer_name);
    for (auto &sp : m_steering_params) {
      sp.m_track_scorer =
          sp.m_track_scorer_name.empty() ? m_default_track_scorer : get_track_scorer(sp.m_track_scorer_name);
    }
  }

  // ============================================================================
  // ConfigJsonPatcher
  // ============================================================================

  ConfigJsonPatcher::ConfigJsonPatcher(bool verbose) : m_verbose(verbose) {}

  ConfigJsonPatcher::~ConfigJsonPatcher() = default;

  std::string ConfigJsonPatcher::get_abs_path() const {
    std::string s;
    s.reserve(64);
    for (auto &p : m_path_stack)
      s += p;
    return s;
  }

  std::string ConfigJsonPatcher::exc_hdr(const char *func) const {
    std::string s;
    s.reserve(128);
    s = "ConfigJsonPatcher";
    if (func) {
      s += "::";
      s += func;
    }
    s += " '";
    s += get_abs_path();
    s += "' ";
    return s;
  }

  template <class T>
  void ConfigJsonPatcher::load(const T &o) {
    m_json = std::make_unique<nlohmann::json>();
    *m_json = o;
    cd_top();
  }
  template void ConfigJsonPatcher::load<IterationsInfo>(const IterationsInfo &o);
  template void ConfigJsonPatcher::load<IterationConfig>(const IterationConfig &o);

  template <class T>
  void ConfigJsonPatcher::save(T &o) {
    from_json(*m_json, o);
  }
  template void ConfigJsonPatcher::save<IterationConfig>(IterationConfig &o);

  // Must not bork the IterationConfig elements of IterationsInfo ... default
  // deserializator apparently reinitializes the vectors with defaults c-tors.
  template <>
  void ConfigJsonPatcher::save<IterationsInfo>(IterationsInfo &o) {
    auto &itc_arr = m_json->at("m_iterations");
    for (int i = 0; i < o.size(); ++i) {
      from_json(itc_arr[i], o[i]);
    }
  }

  void ConfigJsonPatcher::cd(const std::string &path) {
    nlohmann::json::json_pointer jp(path);
    m_json_stack.push_back(m_current);
    m_path_stack.push_back(path);
    m_current = &m_current->at(jp);
  }

  void ConfigJsonPatcher::cd_up(const std::string &path) {
    if (m_json_stack.empty())
      throw std::runtime_error("JSON stack empty on cd_up");

    m_current = m_json_stack.back();
    m_json_stack.pop_back();
    m_path_stack.pop_back();
    if (!path.empty())
      cd(path);
  }

  void ConfigJsonPatcher::cd_top(const std::string &path) {
    m_current = m_json.get();
    m_json_stack.clear();
    m_path_stack.clear();
    if (!path.empty())
      cd(path);
  }

  template <typename T>
  void ConfigJsonPatcher::replace(const std::string &path, T val) {
    nlohmann::json::json_pointer jp(path);
    m_current->at(jp) = val;
  }
  template void ConfigJsonPatcher::replace<int>(const std::string &path, int val);
  template void ConfigJsonPatcher::replace<float>(const std::string &path, float val);
  template void ConfigJsonPatcher::replace<double>(const std::string &path, double val);

  template <typename T>
  void ConfigJsonPatcher::replace(int first, int last, const std::string &path, T val) {
    nlohmann::json::json_pointer jp(path);
    for (int i = first; i <= last; ++i) {
      m_current->at(i).at(jp) = val;
    }
  }
  template void ConfigJsonPatcher::replace<int>(int first, int last, const std::string &path, int val);
  template void ConfigJsonPatcher::replace<float>(int first, int last, const std::string &path, float val);
  template void ConfigJsonPatcher::replace<double>(int first, int last, const std::string &path, double val);

  nlohmann::json &ConfigJsonPatcher::get(const std::string &path) {
    nlohmann::json::json_pointer jp(path);
    return m_current->at(jp);
  }

  int ConfigJsonPatcher::replace(const nlohmann::json &j) {
    if (j.is_null())
      throw std::runtime_error(exc_hdr(__func__) + "null not expected");

    if (j.is_boolean() || j.is_number() || j.is_string()) {
      throw std::runtime_error(exc_hdr(__func__) + "value not expected on this parsing level");
    }

    int n_replaced = 0;

    if (j.is_object()) {
      static const std::regex index_range_re("^\\[(\\d+)..(\\d+)\\]$", std::regex::optimize);

      for (auto &[key, value] : j.items()) {
        std::smatch m;
        std::regex_search(key, m, index_range_re);

        if (m.size() == 3) {
          if (!m_current->is_array())
            throw std::runtime_error(exc_hdr(__func__) + "array range encountered when current json is not an array");
          int first = std::stoi(m.str(1));
          int last = std::stoi(m.str(2));
          for (int i = first; i <= last; ++i) {
            std::string s("/");
            s += std::to_string(i);
            cd(s);
            if (value.is_array()) {
              for (auto &el : value)
                n_replaced += replace(el);
            } else {
              n_replaced += replace(value);
            }
            cd_up();
          }
        } else if (value.is_array() || value.is_object()) {
          std::string s("/");
          s += key;
          cd(s);
          n_replaced += replace(value);
          cd_up();
        } else if (value.is_number() || value.is_boolean() || value.is_string()) {
          std::string s("/");
          s += key;
          nlohmann::json::json_pointer jp(s);
          if (m_current->at(jp) != value) {
            if (m_verbose)
              std::cout << "  " << get_abs_path() << s << ": " << m_current->at(jp) << " -> " << value << "\n";

            m_current->at(jp) = value;
            ++n_replaced;
          }
        } else {
          throw std::runtime_error(exc_hdr(__func__) + "unexpected value type");
        }
      }
    } else if (j.is_array() && j.empty()) {
    } else if (j.is_array()) {
      // Arrays are somewhat tricky.
      // At the moment all elements are expected to be objects.
      //    This means arrays of basic types are not supported (like layer index arrays).
      //    Should not be too hard to add support for this.
      // Now, the objects in the array can be of two kinds:
      // a) Their keys can be json_pointer strings starting with numbers or ranges [i_low..i_high].
      // b) They can be actual elements of the array. In this case we require the length of
      //    the array to be equal to existing length in the configuration.
      // It is not allowed for these two kinds to mix.

      // Determine the kind of array: json_ptr or object

      static const std::regex index_re("^(?:\\[\\d+..\\d+\\]|\\d+(?:/.*)?)$", std::regex::optimize);

      bool has_index = false, has_plain = false;
      for (int i = 0; i < (int)j.size(); ++i) {
        const nlohmann::json &el = j[i];

        if (!el.is_object())
          throw std::runtime_error(exc_hdr(__func__) + "array elements expected to be objects");

        for (nlohmann::json::const_iterator it = el.begin(); it != el.end(); ++it) {
          if (std::regex_search(it.key(), index_re)) {
            has_index = true;
            if (has_plain)
              throw std::runtime_error(exc_hdr(__func__) + "indexed array entry following plain one");
          } else {
            has_plain = true;
            if (has_index)
              throw std::runtime_error(exc_hdr(__func__) + "plain array entry following indexed one");
          }
        }
      }
      if (has_index) {
        for (auto &element : j) {
          n_replaced += replace(element);
        }
      } else {
        if (m_current && !m_current->is_array())
          throw std::runtime_error(exc_hdr(__func__) + "plain array detected when current is not an array");
        if (m_current->size() != j.size())
          throw std::runtime_error(exc_hdr(__func__) + "plain array of different size than at current pos");

        std::string s;
        for (int i = 0; i < (int)j.size(); ++i) {
          s = "/";
          s += std::to_string(i);
          cd(s);
          n_replaced += replace(j[i]);
          cd_up();
        }
      }
    } else {
      throw std::runtime_error(exc_hdr(__func__) + "unexpected json type");
    }

    return n_replaced;
  }

  std::string ConfigJsonPatcher::dump(int indent) { return m_json->dump(indent); }

  // ============================================================================
  // patch_File steering function
  // ============================================================================
  /*
    See example JSON patcher input: "mkFit/config-parse/test.json"

    The file can contain several valid JSON dumps in sequence.

    '/' character can be used to descend more than one level at a time.

    A number can be used to specify an array index. This can be combined with
    the '/' syntax.

    "[first,last]" key (as string) can be used to denote a range of array
    elements. Such a key must not be combined with a '/' syntax.
*/

  namespace {
    // Open file for writing, throw exception on failure.
    void open_ofstream(std::ofstream &ofs, const std::string &fname, const char *pfx = nullptr) {
      ofs.open(fname, std::ofstream::trunc);
      if (!ofs) {
        char m[2048];
        snprintf(m, 2048, "%s%sError opening %s for write: %m", pfx ? pfx : "", pfx ? " " : "", fname.c_str());
        throw std::runtime_error(m);
      }
    }

    // Open file for reading, throw exception on failure.
    void open_ifstream(std::ifstream &ifs, const std::string &fname, const char *pfx = nullptr) {
      ifs.open(fname);
      if (!ifs) {
        char m[2048];
        snprintf(m, 2048, "%s%sError opening %s for read: %m", pfx ? pfx : "", pfx ? " " : "", fname.c_str());
        throw std::runtime_error(m);
      }
    }

    // Skip white-space, return true if more characters are available, false if eof.
    bool skipws_ifstream(std::ifstream &ifs) {
      while (std::isspace(ifs.peek()))
        ifs.get();
      return !ifs.eof();
    }
  }  // namespace

  void ConfigJson::patch_Files(IterationsInfo &its_info,
                               const std::vector<std::string> &fnames,
                               ConfigJsonPatcher::PatchReport *report) {
    ConfigJsonPatcher cjp(m_verbose);
    cjp.load(its_info);

    ConfigJsonPatcher::PatchReport rep;

    for (auto &fname : fnames) {
      std::ifstream ifs;
      open_ifstream(ifs, fname, __func__);

      if (m_verbose) {
        printf("%s begin reading from file %s.\n", __func__, fname.c_str());
      }

      int n_read = 0, n_tot_replaced = 0;
      while (skipws_ifstream(ifs)) {
        nlohmann::json j;
        ifs >> j;
        ++n_read;

        if (m_verbose) {
          std::cout << " Read JSON entity " << n_read << " -- applying patch:\n";
          // std::cout << j.dump(3) << "\n";
        }

        int n_replaced = cjp.replace(j);

        if (m_verbose) {
          std::cout << " Replaced " << n_replaced << " entries.\n";
        }
        cjp.cd_top();
        n_tot_replaced += n_replaced;
      }

      if (m_verbose) {
        printf("%s read %d JSON entities from file %s, replaced %d parameters.\n",
               __func__,
               n_read,
               fname.c_str(),
               n_tot_replaced);
      }

      ifs.close();

      rep.inc_counts(1, n_read, n_tot_replaced);
    }

    if (rep.n_replacements > 0) {
      cjp.save(its_info);
    }

    if (report)
      report->inc_counts(rep);
  }

  std::unique_ptr<IterationConfig> ConfigJson::patchLoad_File(const IterationsInfo &its_info,
                                                              const std::string &fname,
                                                              ConfigJsonPatcher::PatchReport *report) {
    ConfigJsonPatcher::PatchReport rep;

    std::ifstream ifs;
    open_ifstream(ifs, fname, __func__);

    if (m_verbose) {
      printf("%s begin reading from file %s.\n", __func__, fname.c_str());
    }

    if (!skipws_ifstream(ifs))
      throw std::runtime_error("empty file");

    nlohmann::json j;
    ifs >> j;
    int track_algo = j["m_track_algorithm"];

    int iii = -1;
    for (int i = 0; i < its_info.size(); ++i) {
      if (its_info[i].m_track_algorithm == track_algo) {
        iii = i;
        break;
      }
    }
    if (iii == -1)
      throw std::runtime_error("matching IterationConfig not found");

    if (m_verbose) {
      std::cout << " Read JSON entity, Iteration index is " << iii << " -- cloning and applying JSON patch:\n";
    }

    IterationConfig *icp = new IterationConfig(its_info[iii]);
    IterationConfig &ic = *icp;

    ConfigJsonPatcher cjp(m_verbose);
    cjp.load(ic);

    int n_replaced = cjp.replace(j);

    cjp.cd_top();

    if (m_verbose) {
      printf("%s read 1 JSON entity from file %s, replaced %d parameters.\n", __func__, fname.c_str(), n_replaced);
    }

    ifs.close();

    rep.inc_counts(1, 1, n_replaced);

    if (rep.n_replacements > 0) {
      cjp.save(ic);
    }

    if (report)
      report->inc_counts(rep);

    return std::unique_ptr<IterationConfig>(icp);
  }

  std::unique_ptr<IterationConfig> ConfigJson::load_File(const std::string &fname) {
    std::ifstream ifs;
    open_ifstream(ifs, fname, __func__);

    if (m_verbose) {
      printf("%s begin reading from file %s.\n", __func__, fname.c_str());
    }

    if (!skipws_ifstream(ifs))
      throw std::runtime_error("empty file");

    nlohmann::json j;
    ifs >> j;

    if (m_verbose) {
      std::cout << " Read JSON entity, iteration index is " << j["m_iteration_index"] << ", track algorithm is "
                << j["m_track_algorithm"] << ". Instantiating IterationConfig object and over-laying it with JSON.\n";
    }

    IterationConfig *icp = new IterationConfig();

    from_json(j, *icp);

    return std::unique_ptr<IterationConfig>(icp);
  }

  // ============================================================================
  // Save each IterationConfig into a separate json file
  // ============================================================================

  void ConfigJson::save_Iterations(IterationsInfo &its_info,
                                   const std::string &fname_fmt,
                                   bool include_iter_info_preamble) {
    bool has_pct_d = fname_fmt.find("%d") != std::string::npos;
    bool has_pct_s = fname_fmt.find("%s") != std::string::npos;

    assert((has_pct_d || has_pct_s) && "JSON save filename-format must include a %d or %s substring");
    assert(!(has_pct_d && has_pct_s) && "JSON save filename-format must include only one of %d or %s substrings");

    for (int ii = 0; ii < its_info.size(); ++ii) {
      const IterationConfig &itconf = its_info[ii];

      char fname[1024];
      if (has_pct_d)
        snprintf(fname, 1024, fname_fmt.c_str(), ii);
      else
        snprintf(fname, 1024, fname_fmt.c_str(), TrackBase::algoint_to_cstr(itconf.m_track_algorithm));

      std::ofstream ofs;
      open_ofstream(ofs, fname, __func__);

      if (include_iter_info_preamble) {
        ofs << "{ \"m_iterations/" << ii << "\": ";
      }

      nlohmann::ordered_json j;
      to_json(j, itconf);

      ofs << std::setw(1);
      ofs << j;

      if (include_iter_info_preamble) {
        ofs << " }";
      }

      ofs << "\n";
      ofs.close();
    }
  }

  void ConfigJson::dump(IterationsInfo &its_info) {
    nlohmann::ordered_json j = its_info;
    std::cout << j.dump(3) << "\n";
  }

  // ============================================================================
  // Tests for ConfigJson stuff
  // ============================================================================

  void ConfigJson::test_Direct(IterationConfig &it_cfg) {
    using nlohmann::json;

    std::string lojz("/m_select_max_dphi");

    json j = it_cfg;
    std::cout << j.dump(1) << "\n";

    std::cout << "Layer 43, m_select_max_dphi = " << j["/m_layer_configs/43/m_select_max_dphi"_json_pointer] << "\n";
    std::cout << "Patching it to pi ...\n";
    json p = R"([
        { "op": "replace", "path": "/m_layer_configs/43/m_select_max_dphi", "value": 3.141 }
    ])"_json;
    j = j.patch(p);
    std::cout << "Layer 43, m_select_max_dphi = " << j["/m_layer_configs/43/m_select_max_dphi"_json_pointer] << "\n";

    auto &jx = j["/m_layer_configs/60"_json_pointer];
    // jx["m_select_max_dphi"] = 99.876;
    json::json_pointer jp(lojz);
    jx[jp] = 99.876;

    // try loading it back, see what happens to vector m_layer_configs.

    from_json(j, it_cfg);
    printf("Layer 43 : m_select_max_dphi = %f, size_of_layer_vec=%d, m_n_regions=%d, size_of_steering_params=%d\n",
           it_cfg.m_layer_configs[43].m_select_max_dphi,
           (int)it_cfg.m_layer_configs.size(),
           it_cfg.m_n_regions,
           (int)it_cfg.m_steering_params.size());

    printf("Layer 60 : m_select_max_dphi = %f, size_of_layer_vec=%d, m_n_regions=%d, size_of_steering_params=%d\n",
           it_cfg.m_layer_configs[60].m_select_max_dphi,
           (int)it_cfg.m_layer_configs.size(),
           it_cfg.m_n_regions,
           (int)it_cfg.m_steering_params.size());

    // try accessing something that does not exist

    // std::cout << "Non-existent path " << j["/m_layer_configs/143/m_select_max_dphi"_json_pointer] << "\n";

    auto &x = j["/m_layer_configs"_json_pointer];
    std::cout << "Typename /m_layer_configs " << x.type_name() << "\n";
    auto &y = j["/m_layer_configs/143"_json_pointer];
    std::cout << "Typename /m_layer_configs/143 " << y.type_name() << ", is_null=" << y.is_null() << "\n";
  }

  void ConfigJson::test_Patcher(IterationConfig &it_cfg) {
    ConfigJsonPatcher cjp;
    cjp.load(it_cfg);

    std::cout << cjp.dump(1) << "\n";

    {
      cjp.cd("/m_layer_configs/43/m_select_max_dphi");
      std::cout << "Layer 43, m_select_max_dphi = " << cjp.get("") << "\n";
      std::cout << "Setting it to pi ...\n";
      cjp.replace("", 3.141);
      cjp.cd_top();
      std::cout << "Layer 43, m_select_max_dphi = " << cjp.get("/m_layer_configs/43/m_select_max_dphi") << "\n";
    }
    {
      std::cout << "Replacing layer 60 m_select_max_dphi with full path\n";
      cjp.replace("/m_layer_configs/60/m_select_max_dphi", 99.876);
    }
    try {
      std::cout << "Trying to replace an non-existent array entry\n";
      cjp.replace("/m_layer_configs/1460/m_select_max_dphi", 666.666);
    } catch (std::exception &exc) {
      std::cout << "Caugth exception: " << exc.what() << "\n";
    }
    try {
      std::cout << "Trying to replace an non-existent object entry\n";
      cjp.replace("/m_layer_configs/1/moo_select_max_dphi", 666.666);
    } catch (std::exception &exc) {
      std::cout << "Caugth exception: " << exc.what() << "\n";
    }
    {
      std::cout << "Replacing m_select_max_dphi on layers 1 to 3 to 7.7\n";
      cjp.cd("/m_layer_configs");
      cjp.replace(1, 3, "/m_select_max_dphi", 7.7);
      cjp.cd_top();
    }

    // try getting it back into c++, see what happens to vector m_layer_configs.

    cjp.save(it_cfg);

    printf("Layer 43: m_select_max_dphi = %f, size_of_layer_vec=%d, m_n_regions=%d, size_of_steering_params=%d\n",
           it_cfg.m_layer_configs[43].m_select_max_dphi,
           (int)it_cfg.m_layer_configs.size(),
           it_cfg.m_n_regions,
           (int)it_cfg.m_steering_params.size());

    printf("Layer 60: m_select_max_dphi = %f\n", it_cfg.m_layer_configs[60].m_select_max_dphi);
    for (int i = 0; i < 5; ++i)
      printf("Layer %2d: m_select_max_dphi = %f\n", i, it_cfg.m_layer_configs[i].m_select_max_dphi);

    // try accessing something that does not exist

    // std::cout << "Non-existent path " << j["/m_layer_configs/143/m_select_max_dphi"_json_pointer] << "\n";

    auto &j = cjp.get("");

    auto &x = j["/m_layer_configs"_json_pointer];
    std::cout << "Typename /m_layer_configs " << x.type_name() << "\n";
    auto &y = j["/m_layer_configs/143"_json_pointer];
    std::cout << "Typename /m_layer_configs/143 " << y.type_name() << ", is_null=" << y.is_null() << "\n";
  }

}  // namespace mkfit
