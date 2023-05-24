#ifndef RecoTracker_MkFitCore_interface_TrackerInfo_h
#define RecoTracker_MkFitCore_interface_TrackerInfo_h

#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include <string>
#include <vector>

namespace mkfit {

  //==============================================================================
  // WSR -- WithinSensitiveRegion state

  enum WithinSensitiveRegion_e { WSR_Undef = -1, WSR_Inside = 0, WSR_Edge, WSR_Outside, WSR_Failed };

  struct WSR_Result {
    // Could also store XHitSize count equivalent here : 16;
    WithinSensitiveRegion_e m_wsr : 8;
    bool m_in_gap : 8;

    WSR_Result() : m_wsr(WSR_Undef), m_in_gap(false) {}

    WSR_Result(WithinSensitiveRegion_e wsr, bool in_gap) : m_wsr(wsr), m_in_gap(in_gap) {}
  };

  //==============================================================================

  struct ModuleInfo {
    SVector3 pos;
    SVector3 zdir;
    SVector3 xdir;
    unsigned int detid;

    ModuleInfo() = default;
    ModuleInfo(SVector3 p, SVector3 zd, SVector3 xd, unsigned int id) : pos(p), zdir(zd), xdir(xd), detid(id) {}
  };

  //==============================================================================

  class LayerInfo {
    friend class TrackerInfo;

  public:
    enum LayerType_e { Undef = -1, Barrel = 0, EndCapPos = 1, EndCapNeg = 2 };

    LayerInfo() = default;
    LayerInfo(int lid, LayerType_e type) : m_layer_id(lid), m_layer_type(type) {}

    void set_layer_type(LayerType_e t) { m_layer_type = t; }
    void set_limits(float r1, float r2, float z1, float z2);
    void extend_limits(float r, float z);
    void set_r_in_out(float r1, float r2);
    void set_propagate_to(float pto) { m_propagate_to = pto; }
    void set_r_hole_range(float rh1, float rh2);
    void set_q_bin(float qb) { m_q_bin = qb; }
    void set_subdet(int sd) { m_subdet = sd; }
    void set_is_pixel(bool p) { m_is_pixel = p; }
    void set_is_stereo(bool s) { m_is_stereo = s; }

    int layer_id() const { return m_layer_id; }
    LayerType_e layer_type() const { return m_layer_type; }
    float rin() const { return m_rin; }
    float rout() const { return m_rout; }
    float r_mean() const { return 0.5f * (m_rin + m_rout); }
    float zmin() const { return m_zmin; }
    float zmax() const { return m_zmax; }
    float z_mean() const { return 0.5f * (m_zmin + m_zmax); }
    float propagate_to() const { return m_propagate_to; }
    float q_bin() const { return m_q_bin; }

    int subdet() const { return m_subdet; }
    bool is_barrel() const { return m_layer_type == Barrel; }
    bool is_pixel() const { return m_is_pixel; }
    bool is_stereo() const { return m_is_stereo; }

    bool is_within_z_limits(float z) const { return z > m_zmin && z < m_zmax; }
    bool is_within_r_limits(float r) const { return r > m_rin && r < m_rout; }
    bool is_within_q_limits(float q) const { return is_barrel() ? is_within_z_limits(q) : is_within_r_limits(q); }

    bool is_in_r_hole(float r) const { return m_has_r_range_hole ? is_in_r_hole_no_check(r) : false; }

    WSR_Result is_within_z_sensitive_region(float z, float dz) const {
      if (z > m_zmax + dz || z < m_zmin - dz)
        return WSR_Result(WSR_Outside, false);
      if (z < m_zmax - dz && z > m_zmin + dz)
        return WSR_Result(WSR_Inside, false);
      return WSR_Result(WSR_Edge, false);
    }

    WSR_Result is_within_r_sensitive_region(float r, float dr) const {
      if (r > m_rout + dr || r < m_rin - dr)
        return WSR_Result(WSR_Outside, false);
      if (r < m_rout - dr && r > m_rin + dr) {
        if (m_has_r_range_hole) {
          if (r < m_hole_r_max - dr && r > m_hole_r_min + dr)
            return WSR_Result(WSR_Outside, true);
          if (r < m_hole_r_max + dr && r > m_hole_r_min - dr)
            return WSR_Result(WSR_Edge, true);
        }
        return WSR_Result(WSR_Inside, false);
      }
      return WSR_Result(WSR_Edge, false);
    }

    void print_layer() const;

    // module & detid interface
    void reserve_modules(int nm) { m_modules.reserve(nm); }
    unsigned int register_module(ModuleInfo&& mi) {
      unsigned int pos = m_modules.size();
      m_modules.emplace_back(mi);
      m_detid2sid[mi.detid] = pos;
      return pos;
    }
    unsigned int shrink_modules() {
      m_modules.shrink_to_fit();
      return m_modules.size() - 1;
    }

    unsigned int short_id(unsigned int detid) const { return m_detid2sid.at(detid); }
    int n_modules() const { return m_modules.size(); }
    const ModuleInfo& module_info(unsigned int sid) const { return m_modules[sid]; }

  private:
    bool is_in_r_hole_no_check(float r) const { return r > m_hole_r_min && r < m_hole_r_max; }

    int m_layer_id = -1;
    LayerType_e m_layer_type = Undef;
    int m_subdet = -1;  // sub-detector id, not used in core mkFit

    float m_rin = 0, m_rout = 0, m_zmin = 0, m_zmax = 0;
    float m_propagate_to = 0;

    float m_q_bin = 0;                         // > 0 - bin width, < 0 - number of bins
    float m_hole_r_min = 0, m_hole_r_max = 0;  // This could be turned into std::function when needed.
    bool m_has_r_range_hole = false;
    bool m_is_stereo = false;
    bool m_is_pixel = false;

    std::unordered_map<unsigned int, unsigned int> m_detid2sid;
    std::vector<ModuleInfo> m_modules;
  };

  //==============================================================================

  template <typename T>
  class rectvec {
  public:
    rectvec(int n1=0, int n2=0) : m_n1(n1), m_n2(n2), m_vec(n1 * n2) {}

    void rerect(int n1, int n2) {
      m_n1 = n1;
      m_n2 = n2;
      m_vec.resize(n1 * n2);
    }

    const T& operator()(int i1, int i2) const { return m_vec[i1 * m_n2 + i2]; }
    T& operator()(int i1, int i2) { return m_vec[i1 * m_n2 + i2]; }

    const T* operator[](int i1) const { return &m_vec[i1 * m_n2]; }
    T* operator[](int i1) { return &m_vec[i1 * m_n2]; }

    const std::vector<T>& vector() const { return m_vec; }
    std::vector<T>& vector() { return m_vec; }

    int n1() const { return m_n1; }
    int n2() const { return m_n2; }
    bool check_idcs(int i1, int i2) const { return i1 >= 0 && i1 < m_n1 && i2 >= 0 && i2 < m_n2; }

  private:
    int m_n1, m_n2;
    std::vector<T> m_vec;
  };

  class TrackerInfo {
  public:
    enum EtaRegion {
      Reg_Begin = 0,
      Reg_Endcap_Neg = 0,
      Reg_Transition_Neg,
      Reg_Barrel,
      Reg_Transition_Pos,
      Reg_Endcap_Pos,
      Reg_End,
      Reg_Count = Reg_End
    };
    struct Material {
      float bbxi{0}, radl{0};
    };

    void reserve_layers(int n_brl, int n_ec_pos, int n_ec_neg);
    void create_layers(int n_brl, int n_ec_pos, int n_ec_neg);
    LayerInfo& new_barrel_layer();
    LayerInfo& new_ecap_pos_layer();
    LayerInfo& new_ecap_neg_layer();

    int n_layers() const { return m_layers.size(); }
    const LayerInfo& layer(int l) const { return m_layers[l]; }
    LayerInfo& layer_nc(int l) { return m_layers[l]; }

    int n_total_modules() const;

    const LayerInfo& operator[](int l) const { return m_layers[l]; }

    const LayerInfo& outer_barrel_layer() const { return m_layers[m_barrel.back()]; }

    const std::vector<int>& barrel_layers() const { return m_barrel; }
    const std::vector<int>& endcap_pos_layers() const { return m_ecap_pos; }
    const std::vector<int>& endcap_neg_layers() const { return m_ecap_neg; }

    const PropagationConfig& prop_config() const { return m_prop_config; }
    PropagationConfig& prop_config_nc() { return m_prop_config; }

    void write_bin_file(const std::string& fname) const;
    void read_bin_file(const std::string& fname);
    void print_tracker(int level) const;

    void create_material(int nBinZ, float rngZ, int nBinR, float rngR);
    int mat_nbins_z() const { return m_mat_vec.n1(); }
    int mat_nbins_r() const { return m_mat_vec.n2(); }
    float mat_range_z() const { return m_mat_range_z; }
    float mat_range_r() const { return m_mat_range_r; }
    int mat_bin_z(float z) const { return z * m_mat_fac_z; }
    int mat_bin_r(float r) const { return r * m_mat_fac_r; }
    bool check_bins(int bz, int br) const { return m_mat_vec.check_idcs(bz, br); }

    float material_bbxi(int binZ, int binR) const { return m_mat_vec(binZ, binR).bbxi; }
    float material_radl(int binZ, int binR) const { return m_mat_vec(binZ, binR).radl; }
    float& material_bbxi(int binZ, int binR) { return m_mat_vec(binZ, binR).bbxi; }
    float& material_radl(int binZ, int binR) { return m_mat_vec(binZ, binR).radl; }

    Material material_checked(float z, float r) const {
      const int zbin = mat_bin_z(z), rbin = mat_bin_r(r);
      return check_bins(zbin, rbin) ? m_mat_vec(zbin, rbin) : Material();
    }

  private:
    int new_layer(LayerInfo::LayerType_e type);

    std::vector<LayerInfo> m_layers;

    std::vector<int> m_barrel;
    std::vector<int> m_ecap_pos;
    std::vector<int> m_ecap_neg;

    float m_mat_range_z, m_mat_range_r;
    float m_mat_fac_z, m_mat_fac_r;
    rectvec<Material> m_mat_vec;

    PropagationConfig m_prop_config;
  };

}  // end namespace mkfit
#endif
