#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"

#include <cassert>
#include <cstring>

namespace mkfit {

  //==============================================================================
  // PropagationConfig
  //==============================================================================

  void PropagationConfig::apply_tracker_info(const TrackerInfo* ti) {
    finding_inter_layer_pflags.tracker_info = ti;
    finding_intra_layer_pflags.tracker_info = ti;
    backward_fit_pflags.tracker_info = ti;
    forward_fit_pflags.tracker_info = ti;
    seed_fit_pflags.tracker_info = ti;
    pca_prop_pflags.tracker_info = ti;
  }

  //==============================================================================
  // LayerInfo
  //==============================================================================

  void LayerInfo::set_limits(float r1, float r2, float z1, float z2) {
    m_rin = r1;
    m_rout = r2;
    m_zmin = z1;
    m_zmax = z2;
  }

  void LayerInfo::extend_limits(float r, float z) {
    if (z > m_zmax)
      m_zmax = z;
    if (z < m_zmin)
      m_zmin = z;
    if (r > m_rout)
      m_rout = r;
    if (r < m_rin)
      m_rin = r;
  }

  void LayerInfo::set_r_in_out(float r1, float r2) {
    m_rin = r1;
    m_rout = r2;
  }

  void LayerInfo::set_r_hole_range(float rh1, float rh2) {
    m_has_r_range_hole = true;
    m_hole_r_min = rh1;
    m_hole_r_max = rh2;
  }

  void LayerInfo::print_layer() const {
    // clang-format off
    printf("Layer %2d  r(%7.4f, %7.4f) z(% 9.4f, % 9.4f)"
           " is_brl=%d, is_pix=%d, is_stereo=%d, has_charge=%d, q_bin=%.2f\n",
           m_layer_id, m_rin, m_rout, m_zmin, m_zmax,
           is_barrel(), m_is_pixel, m_is_stereo, m_has_charge, m_q_bin);
    if (m_has_r_range_hole)
      printf("          has_r_range_hole: %.2f -> %.2f, dr: %f\n", m_hole_r_min, m_hole_r_max, m_hole_r_max - m_hole_r_min);
    // clang-format on
  }

  //==============================================================================
  // TrackerInfo
  //==============================================================================

  void TrackerInfo::reserve_layers(int n_brl, int n_ec_pos, int n_ec_neg) {
    m_layers.reserve(n_brl + n_ec_pos + n_ec_neg);
    m_barrel.reserve(n_brl);
    m_ecap_pos.reserve(n_ec_pos);
    m_ecap_neg.reserve(n_ec_neg);
  }

  void TrackerInfo::create_layers(int n_brl, int n_ec_pos, int n_ec_neg) {
    reserve_layers(n_brl, n_ec_pos, n_ec_neg);
    for (int i = 0; i < n_brl; ++i)
      new_barrel_layer();
    for (int i = 0; i < n_ec_pos; ++i)
      new_ecap_pos_layer();
    for (int i = 0; i < n_ec_neg; ++i)
      new_ecap_neg_layer();
  }

  int TrackerInfo::new_layer(LayerInfo::LayerType_e type) {
    int l = (int)m_layers.size();
    m_layers.emplace_back(LayerInfo(l, type));
    return l;
  }

  LayerInfo& TrackerInfo::new_barrel_layer() {
    m_barrel.push_back(new_layer(LayerInfo::Barrel));
    return m_layers.back();
  }

  LayerInfo& TrackerInfo::new_ecap_pos_layer() {
    m_ecap_pos.push_back(new_layer(LayerInfo::EndCapPos));
    return m_layers.back();
  }

  LayerInfo& TrackerInfo::new_ecap_neg_layer() {
    m_ecap_neg.push_back(new_layer(LayerInfo::EndCapNeg));
    return m_layers.back();
  }

  int TrackerInfo::n_total_modules() const {
    int nm = 0;
    for (auto& l : m_layers)
      nm += l.n_modules();
    return nm;
  }

  //==============================================================================
  // Material

  void TrackerInfo::create_material(int nBinZ, float rngZ, int nBinR, float rngR) {
    m_mat_range_z = rngZ;
    m_mat_range_r = rngR;
    m_mat_fac_z = nBinZ / m_mat_range_z;
    m_mat_fac_r = nBinR / m_mat_range_r;

    m_mat_vec.rerect(nBinZ, nBinR);
  }

  //==============================================================================

  namespace {
    struct GeomFileHeader {
      int f_magic = s_magic;
      int f_format_version = s_version;
      int f_sizeof_trackerinfo = sizeof(TrackerInfo);
      int f_sizeof_layerinfo = sizeof(LayerInfo);
      int f_sizeof_moduleinfo = sizeof(ModuleInfo);
      int f_sizeof_moduleshape = sizeof(ModuleShape);
      int f_n_layers = -1;

      GeomFileHeader() = default;

      constexpr static int s_magic = 0xB00F;
      constexpr static int s_version = 3;
    };

    template <typename T>
    int write_std_vec(FILE* fp, const std::vector<T>& vec, int el_size = 0) {
      int n = vec.size();
      fwrite(&n, sizeof(int), 1, fp);
      if (el_size == 0) {
        fwrite(vec.data(), sizeof(T), n, fp);
      } else {
        for (int i = 0; i < n; ++i)
          fwrite(&vec[i], el_size, 1, fp);
      }
      return n;
    }

    template <typename T>
    int read_std_vec(FILE* fp, std::vector<T>& vec, int el_size = 0) {
      int n;
      fread(&n, sizeof(int), 1, fp);
      vec.resize(n);
      if (el_size == 0) {
        fread(vec.data(), sizeof(T), n, fp);
      } else {
        for (int i = 0; i < n; ++i)
          fread(&vec[i], el_size, 1, fp);
      }
      return n;
    }

    void assert_sizeof_match(int size_on_file, int size_of_class, const char* class_name) {
      if (size_on_file != size_of_class) {
        fprintf(stderr,
                "sizeof(%s) on file (%d) different from current value (%d).\n",
                class_name,
                size_on_file,
                size_of_class);
        throw std::runtime_error("class sizeof mismatch for " + std::string(class_name));
      }
    }
  }  // namespace

  void TrackerInfo::write_bin_file(const std::string& fname) const {
    FILE* fp = fopen(fname.c_str(), "w");
    if (!fp) {
      fprintf(stderr,
              "TrackerInfo::write_bin_file error opening file '%s', errno=%d: '%s'",
              fname.c_str(),
              errno,
              strerror(errno));
      throw std::runtime_error("Filed opening file in TrackerInfo::write_bin_file");
    }
    GeomFileHeader fh;
    fh.f_n_layers = n_layers();
    fwrite(&fh, sizeof(GeomFileHeader), 1, fp);

    write_std_vec(fp, m_layers, (int)offsetof(LayerInfo, m_final_member_for_streaming));
    write_std_vec(fp, m_barrel);
    write_std_vec(fp, m_ecap_pos);
    write_std_vec(fp, m_ecap_neg);

    for (int l = 0; l < fh.f_n_layers; ++l) {
      write_std_vec(fp, m_layers[l].m_modules);
      write_std_vec(fp, m_layers[l].m_shapes);
    }

    fwrite(&m_mat_range_z, 4 * sizeof(float), 1, fp);
    fwrite(&m_mat_vec, 2 * sizeof(int), 1, fp);
    write_std_vec(fp, m_mat_vec.vector());

    fclose(fp);
  }

  void TrackerInfo::read_bin_file(const std::string& fname) {
    FILE* fp = fopen(fname.c_str(), "r");
    if (!fp) {
      fprintf(stderr,
              "TrackerInfo::read_bin_file error opening file '%s', errno=%d: '%s'\n",
              fname.c_str(),
              errno,
              strerror(errno));
      throw std::runtime_error("Failed opening file in TrackerInfo::read_bin_file");
    }
    GeomFileHeader fh;
    fread(&fh, sizeof(GeomFileHeader), 1, fp);

    if (fh.f_magic != GeomFileHeader::s_magic) {
      fprintf(stderr, "Incompatible input file (wrong magick).\n");
      throw std::runtime_error("Filed opening file in TrackerInfo::read_bin_file");
    }
    if (fh.f_format_version != GeomFileHeader::s_version) {
      fprintf(stderr,
              "Unsupported file version %d. Supported version is %d.\n",
              fh.f_format_version,
              GeomFileHeader::s_version);
      throw std::runtime_error("Unsupported file version in TrackerInfo::read_bin_file");
    }
    assert_sizeof_match(fh.f_sizeof_trackerinfo, sizeof(TrackerInfo), "TrackerInfo");
    assert_sizeof_match(fh.f_sizeof_layerinfo, sizeof(LayerInfo), "LayerInfo");
    assert_sizeof_match(fh.f_sizeof_moduleinfo, sizeof(ModuleInfo), "ModuleInfo");
    assert_sizeof_match(fh.f_sizeof_moduleshape, sizeof(ModuleShape), "ModuleShape");

    printf("Opened TrackerInfoGeom file '%s', format version %d, n_layers %d\n",
           fname.c_str(),
           fh.f_format_version,
           fh.f_n_layers);

    read_std_vec(fp, m_layers, (int)offsetof(LayerInfo, m_final_member_for_streaming));
    read_std_vec(fp, m_barrel);
    read_std_vec(fp, m_ecap_pos);
    read_std_vec(fp, m_ecap_neg);

    for (int l = 0; l < fh.f_n_layers; ++l) {
      LayerInfo& li = m_layers[l];
      int nm = read_std_vec(fp, li.m_modules);

      li.m_detid2sid.clear();
      for (int m = 0; m < nm; ++m) {
        li.m_detid2sid.insert({li.m_modules[m].detid, m});
      }

      read_std_vec(fp, li.m_shapes);
    }

    fread(&m_mat_range_z, 4 * sizeof(float), 1, fp);
    fread(&m_mat_vec, 2 * sizeof(int), 1, fp);
    read_std_vec(fp, m_mat_vec.vector());

    fclose(fp);
  }

  void TrackerInfo::print_tracker(int level, int precision) const {
    // clang-format off
    if (level > 0) {
      for (int i = 0; i < n_layers(); ++i) {
        const LayerInfo& li = layer(i);
        li.print_layer();
        if (level > 1) {
          printf("  Detailed module list N=%d, N_shapes=%d\n", li.n_modules(), li.n_shapes());
          for (int j = 0; j < li.n_shapes(); ++j) {
            const ModuleShape &ms = li.module_shape(j);
            const int w = precision + 1;
            printf("    Shape id=%u: dx1=%.*f, dx2=%.*f, dy=%.*f, dz=%.*f\n",
                   j, w, ms.dx1, w, ms.dx2, w, ms.dy, w, ms.dz);
          }
          for (int j = 0; j < li.n_modules(); ++j) {
            const ModuleInfo& mi = li.module_info(j);
            auto* p = mi.pos.Array();
            auto* z = mi.zdir.Array();
            auto* x = mi.xdir.Array();
            const int w = precision;
            printf("    Module id=%u: detid=0x%x shapeid=%u pos=%.*f,%.*f,%.*f, "
                   "norm=%.*f,%.*f,%.*f, phi=%.*f,%.*f,%.*f\n",
                   j, mi.detid, mi.shapeid, w, p[0], w, p[1], w, p[2],
                   w, z[0], w, z[1], w, z[2], w, x[0], w, x[1], w, x[2]);
          }
          printf("\n");
        }
      }
    }
    // clang-format on
  }
}  // end namespace mkfit
