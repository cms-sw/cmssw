#ifndef L1Trigger_Phase2L1ParticleFlow_METEmulator_h
#define L1Trigger_Phase2L1ParticleFlow_METEmulator_h

#include "DataFormats/L1TParticleFlow/interface/jets.h"
#include "DataFormats/L1TParticleFlow/interface/sums.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include "nlohmann/json.hpp"

#ifdef CMSSW_GIT_HASH
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#endif

#ifndef CMSSW_GIT_HASH
#include "hls_math.h"
#endif

#include <vector>
#include <array>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "ap_int.h"
#include "ap_fixed.h"

namespace L1METEmu {
  // Define Data types for P2 L1 MET Emulator
  typedef l1ct::pt_t pt_t;
  typedef l1ct::glbphi_t phi_t;
  typedef l1ct::glbeta_t eta_t;
  typedef l1ct::dpt_t pxy_t;
  typedef l1ct::Sum Met;

  typedef ap_fixed<22, 12> proj_t;
  typedef ap_fixed<32, 22> proj2_t;
  typedef ap_fixed<32, 2> poly_t;

  struct Particle_xy {
    // 44bits
    proj_t hwPx;
    proj_t hwPy;
  };

  struct Poly2_param {
    std::vector<poly_t> c0, c1, c2, s0, s1, s2;
    std::vector<phi_t> phi_edges;
    unsigned int phi_bins = 0;
  };

  struct Poly2_Path {
    std::string path = "L1Trigger/Phase2L1ParticleFlow/data/met/l1met_ptphi2pxpy_poly2_v1.json";
  };

  inline Poly2_Path& poly2_path_config() {
    static Poly2_Path met_p;
    return met_p;
  }

  inline void SetPoly2File(std::string met_p) { poly2_path_config().path = std::move(met_p); }

  inline const Poly2_param& Get_Poly2_param() {
    static Poly2_param P = [] {
      Poly2_param t{};
      std::string path = poly2_path_config().path;

#ifdef CMSSW_GIT_HASH
      edm::FileInPath f(path);
      std::ifstream in(f.fullPath());
      if (!in) {
        throw cms::Exception("FileNotFound") << f.fullPath();
      }
#else
      path = "l1met_ptphi2pxpy_poly2_v1.json";  // For HLS Emulator
      std::ifstream in(path);
      if (!in) {
        throw std::runtime_error(std::string("File not found: ") + path);
      }
#endif

      nlohmann::json j;
      in >> j;

      unsigned int N = j["phi_bins"].get<unsigned int>();
      t.phi_bins = N;

      t.c0.resize(N);
      t.c1.resize(N);
      t.c2.resize(N);
      t.s0.resize(N);
      t.s1.resize(N);
      t.s2.resize(N);
      t.phi_edges.resize(N + 1);

      for (unsigned int i = 0; i < N; ++i) {
        t.c0[i] = poly_t(j["cos"]["par0"][i].get<double>());
        t.c1[i] = poly_t(j["cos"]["par1"][i].get<double>());
        t.c2[i] = poly_t(j["cos"]["par2"][i].get<double>());
        t.s0[i] = poly_t(j["sin"]["par0"][i].get<double>());
        t.s1[i] = poly_t(j["sin"]["par1"][i].get<double>());
        t.s2[i] = poly_t(j["sin"]["par2"][i].get<double>());
      }
      for (unsigned int i = 0; i < N + 1; ++i) {
        t.phi_edges[i] = l1ct::Scales::makeGlbPhi(j["phi_edges"][i].get<double>() * M_PI);
      }
      return t;
    }();
    return P;
  }

  inline Particle_xy Get_xy(const l1ct::pt_t hwPt, const l1ct::glbphi_t hwPhi) {
    /*
      Convert pt, phi to px, py
      - Use 2nd order Polynomial interpolation for cos, sin
      - Divide the sine and cosine value from -pi to pi into 16 parts
      - Fitting the value with 2nd order function
    */

    const auto& P = L1METEmu::Get_Poly2_param();
    int phibin = 0;

    for (unsigned int i = 0; i < P.phi_bins; i++) {
      if (hwPhi >= P.phi_edges[i] && hwPhi < P.phi_edges[i + 1]) {
        phibin = i;
        break;
      }
    }
    // Handle the edge case where hwPhi is exactly equal to the last bin edge
    if (hwPhi == P.phi_edges[P.phi_bins]) {
      phibin = P.phi_bins - 1;
    }

    Particle_xy proj_xy;

    poly_t cos_var = P.c0[phibin] + P.c1[phibin] * (hwPhi - P.phi_edges[phibin]) +
                     P.c2[phibin] * (hwPhi - P.phi_edges[phibin]) * (hwPhi - P.phi_edges[phibin]);
    poly_t sin_var = P.s0[phibin] + P.s1[phibin] * (hwPhi - P.phi_edges[phibin]) +
                     P.s2[phibin] * (hwPhi - P.phi_edges[phibin]) * (hwPhi - P.phi_edges[phibin]);

    proj_xy.hwPx = hwPt * cos_var;
    proj_xy.hwPy = hwPt * sin_var;

    return proj_xy;
  }

  inline void Sum_Particles(const std::vector<Particle_xy>& particles_xy, Particle_xy& met_xy) {
    met_xy.hwPx = 0;
    met_xy.hwPy = 0;

    for (unsigned int i = 0; i < particles_xy.size(); ++i) {
      met_xy.hwPx -= particles_xy[i].hwPx;
      met_xy.hwPy -= particles_xy[i].hwPy;
    }
  }

  inline void pxpy_to_ptphi(const Particle_xy met_xy, l1ct::Sum& hls_met) {
    // convert x, y coordinate to pt, phi coordinate using math library
    hls_met.clear();

#ifdef CMSSW_GIT_HASH
    hls_met.hwPt = hypot(met_xy.hwPx.to_float(), met_xy.hwPy.to_float());
    hls_met.hwPhi = phi_t(ap_fixed<26, 11>(atan2(met_xy.hwPy.to_float(), met_xy.hwPx.to_float())) *
                          ap_fixed<26, 11>(229.29936));  // Scale for L1 phi value (720 / M_PI)
#else
    hls_met.hwPt = hls::hypot(met_xy.hwPx, met_xy.hwPy);
    hls_met.hwPhi = phi_t(ap_fixed<26, 11>(hls::atan2(met_xy.hwPy, met_xy.hwPx)) * ap_fixed<26, 11>(229.29936));
#endif
  }

  inline void met_format(const l1ct::Sum d, ap_uint<l1gt::Sum::BITWIDTH>& q) {
    // Change output formats to GT formats
    q = d.toGT().pack();
  }

}  // namespace L1METEmu

inline void puppimet_emu(const std::vector<l1ct::PuppiObjEmu>& particles, l1ct::Sum& out_met) {
  std::vector<L1METEmu::Particle_xy> particles_xy;

  for (unsigned int i = 0; i < particles.size(); i++) {
    L1METEmu::Particle_xy each_particle_xy = L1METEmu::Get_xy(particles[i].hwPt, particles[i].hwPhi);
    particles_xy.push_back(each_particle_xy);
  }

  L1METEmu::Particle_xy met_xy;
  L1METEmu::Sum_Particles(particles_xy, met_xy);
  L1METEmu::pxpy_to_ptphi(met_xy, out_met);
}

#endif
