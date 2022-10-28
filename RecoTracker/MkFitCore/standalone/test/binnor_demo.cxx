#include <random>
#include <chrono>

#include "../../interface/radix_sort.h"
#include "../../src/radix_sort.cc"
#include "../../interface/binnor.h"

// build as:
// c++ -o binnor_demo -O3 -mavx2 -std=c++17 -I../../../.. binnor_demo.cxx

bool print_axis_binnor = true;
bool print_bin_contents = true;

void run_scan();
void run_test(const int NLoop, int NN, const bool use_radix = true, const bool keep_cons = false);

using namespace mkfit;

int main()
{
  bool use_radix = true;
  bool keep_cons = false;
  run_test(100, 10000, use_radix, keep_cons);

  // Run scan over low N-elements to see where radix starts to make more sense.
  // NOTE: In binnor.h, finalize_registration(), comment-out the existing heuristics
  // that uses std::sort over radix when NN < 256 (128 with keep_cons).
  // run_scan();

}

// -------------------------------------------------------------

void run_scan()
{
  print_axis_binnor = false;
  print_bin_contents = false;

  const bool radix[] = { true, true, false, false };
  const bool keepc[] = { false, true, false, true };

  for (int t = 0; t < 4; ++t) {
    printf("RADIX=%s  KEEP-CONS=%s\n", radix[t] ? "true" : "false", keepc[t] ? "true" : "false");
    for (int i = 100; i <= 400; i += 10) {
      run_test(100000, i, radix[t], keepc[t]);
    }
    printf("\n");
  }
}

// -------------------------------------------------------------

void run_test(const int NLoop, int NN, const bool use_radix, const bool keep_cons) {
  constexpr float PI = 3.14159265358979323846;
  constexpr float TwoPI = 6.28318530717958647692;
  constexpr float PIOver2 = PI / 2.0f;
  constexpr float PIOver4 = PI / 4.0f;

  axis_pow2_u1<float, unsigned short, 16, 8> phi(-PI, PI);
  axis<float, unsigned short, 16, 8> eta(-2.6, 2.6, 20u);
  binnor<unsigned int, decltype(phi), decltype(eta), 18, 14> b(phi, eta, use_radix, keep_cons);

  if (print_axis_binnor) {
    printf("Axis phi: M-bits=%d, N-bits=%d  Masks M:0x%x N:0x%x\n", phi.c_M, phi.c_N, phi.c_M_mask, phi.c_N_mask);
    printf("Axis eta: M-bits=%d, N-bits=%d  m_bins=%d n_bins=%d\n", eta.c_M, eta.c_N, eta.size_of_M(), eta.size_of_N());
    printf("Have binnor, size of vec = %zu, sizeof(C_pair) = %d\n", b.m_bins.size(), sizeof(decltype(b)::C_pair));
  }
  /*
    for (float p = -TwoPI; p < TwoPI; p += TwoPI / 15.4f) {
        printf("  phi=%-9f m=%5d n=%3d m2n=%3d n_safe=%3d\n", p,
               phi.from_R_to_M_bin(p), phi.from_R_to_N_bin(p),
               phi.from_M_bin_to_N_bin( phi.from_R_to_M_bin(p) ),
               phi.from_R_to_N_bin_safe(p) );
    }
    */

  std::mt19937 rnd(std::random_device{}());
  std::uniform_real_distribution<float> d_phi(-PI, PI);
  std::uniform_real_distribution<float> d_eta(-2.55, 2.55);

  struct track {
    float phi, eta;
  };
  std::vector<track> tracks;
  tracks.reserve(NN);
  for (int i = 0; i < NN; ++i) {
    tracks.push_back({d_phi(rnd), d_eta(rnd)});
    // printf("made track %3d:  phi=%f  eta=%f\n", i, tracks.back().phi, tracks.back().eta);
  }

  std::chrono::duration<long long, std::nano> duration, fill_duration, sort_duration;
  duration = fill_duration = sort_duration = std::chrono::nanoseconds::zero();

  for (int n_loop = 0; n_loop < NLoop; ++n_loop) {
    b.reset_contents(false); // do not shrink the vectors

    auto start = std::chrono::high_resolution_clock::now();

    b.begin_registration(NN);  // NN optional, reserves construction vector

    for (int i = 0; i < NN; ++i) {
      b.register_entry(tracks[i].phi, tracks[i].eta);
    }

    auto reg_start = std::chrono::high_resolution_clock::now();

    b.finalize_registration();

    auto stop = std::chrono::high_resolution_clock::now();

    duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    fill_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(reg_start - start);
    sort_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - reg_start);
  }

  // for (int i = 0; i < NN; ++i)
  // {
  //     const track &t = tracks[ b.m_ranks[i] ];
  //     printf("%3d  %3d  phi=%f  eta=%f\n", i, b.m_ranks[i], t.phi, t.eta);
  // }

  if (print_bin_contents) {
    printf("\n\n--- Single bin access for (phi, eta) = (0,0):\n\n");
    auto nbin = b.get_n_bin(0.f, 0.f);
    auto cbin = b.get_content(0.f, 0.f);
    printf("For (phi 0, eta 0; %u, %u) got first %d, count %d\n", nbin.bin1(), nbin.bin2(), cbin.first, cbin.count);
    for (auto i = cbin.first; i < cbin.first + cbin.count; ++i) {
      const track &t = tracks[b.m_ranks[i]];
      printf("%3d  %3d  phi=%f  eta=%f\n", i, b.m_ranks[i], t.phi, t.eta);
    }

    printf("\n\n--- Range access for phi=[(-PI+0.02 +- 0.1], eta=[1.3 +- .2]:\n\n");
    auto phi_rng = phi.from_R_rdr_to_N_bins(-PI + 0.02, 0.1);
    auto eta_rng = eta.from_R_rdr_to_N_bins(1.3, .2);
    printf("phi bin range: %u, %u; eta %u, %u\n", phi_rng.begin, phi_rng.end, eta_rng.begin, eta_rng.end);
    for (auto i_phi = phi_rng.begin; i_phi != phi_rng.end; i_phi = phi.next_N_bin(i_phi)) {
      for (auto i_eta = eta_rng.begin; i_eta != eta_rng.end; i_eta = eta.next_N_bin(i_eta)) {
        printf(" at i_phi=%u, i_eta=%u\n", i_phi, i_eta);
        const auto cbin = b.get_content(i_phi, i_eta);
        for (auto i = cbin.begin(); i < cbin.end(); ++i) {
          const track &t = tracks[b.m_ranks[i]];
          printf("   %3d  %3d  phi=%f  eta=%f\n", i, b.m_ranks[i], t.phi, t.eta);
        }
      }
    }
    printf("\n");
  }

  printf("Processing time for %d times %d points: %f sec (filling %f sec, sort + binning %f sec)\n",
         NLoop, NN,
         1.e-9 * duration.count(),
         1.e-9 * fill_duration.count(),
         1.e-9 * sort_duration.count());
}
