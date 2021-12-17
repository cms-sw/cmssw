#include "binnor.h"
#include <random>

// build as:
// c++ -o binnor_demo -std=c++17 binnor_demo.cxx

using namespace mkfit;

int main()
{
    constexpr float    PI    = 3.14159265358979323846;
    constexpr float TwoPI    = 6.28318530717958647692;
    constexpr float PIOver2  = PI / 2.0f;
    constexpr float PIOver4  = PI / 4.0f;

    axis_pow2_u1<float, unsigned short, 16, 8> phi(-PI, PI);

    printf("Axis phi: M-bits=%d, N-bits=%d  Masks M:0x%x N:0x%x\n",
           phi.c_M, phi.c_N, phi.c_M_mask, phi.c_N_mask);

    /*
    for (float p = -TwoPI; p < TwoPI; p += TwoPI / 15.4f) {
        printf("  phi=%-9f m=%5d n=%3d m2n=%3d n_safe=%3d\n", p,
               phi.R_to_M_bin(p), phi.R_to_N_bin(p),
               phi.M_bin_to_N_bin( phi.R_to_M_bin(p) ),
               phi.R_to_N_bin_safe(p) );
    }
    */

    axis<float, unsigned short, 12, 6> eta(-2.6, 2.6, 20u);

    printf("Axis eta: M-bits=%d, N-bits=%d    n_bins=%d\n",
           eta.c_M, eta.c_N, eta.m_n_bins);

    binnor<unsigned int, decltype(phi), decltype(eta), 24, 8> b(phi, eta);

    // typedef typeof(b) type_b;
    printf("Have binnor, size of vec = %zu, sizeof(C_pair) = %d\n",
           b.m_bins.size(), sizeof( decltype(b)::C_pair) );

    std::mt19937 rnd(std::random_device{}());
    std::uniform_real_distribution<float> d_phi(-PI, PI);
    std::uniform_real_distribution<float> d_eta(-2.55, 2.55);

    const int NN = 100000;

    struct track { float phi, eta; };
    std::vector<track> tracks;
    tracks.reserve(NN);

    b.begin_registration(NN); // optional, reserves construction vector

    for (int i = 0; i < NN; ++i)
    {
        tracks.push_back( { d_phi(rnd), d_eta(rnd) } );
        b.register_entry(tracks.back().phi, tracks.back().eta);
        // printf("made track %3d:  phi=%f  eta=%f\n", i, tracks.back().phi, tracks.back().eta);
    }

    b.finalize_registration();

    // for (int i = 0; i < NN; ++i)
    // {
    //     const track &t = tracks[ b.m_ranks[i] ];
    //     printf("%3d  %3d  phi=%f  eta=%f\n", i, b.m_ranks[i], t.phi, t.eta);
    // }

    printf("\n\n--- Single bin access:\n\n");
    auto nbin = b.get_n_bin(0.f, 0.f);
    auto cbin = b.get_content(0.f, 0.f);
    printf("For (phi 0, eta 0; %u, %u) got first %d, count %d\n", nbin.bin1, nbin.bin2, cbin.first, cbin.count);
    for (auto i = cbin.first; i < cbin.first + cbin.count; ++i) {
        const track &t = tracks[ b.m_ranks[i] ];
        printf("%3d  %3d  phi=%f  eta=%f\n", i, b.m_ranks[i], t.phi, t.eta);
    }

    printf("\n\n--- Range access:\n\n");
    auto phi_rng = phi.Rrdr_to_N_bins(-PI+0.02, 0.1);
    auto eta_rng = eta.Rrdr_to_N_bins(1.3, .2);
    printf("phi bin range: %u, %u; eta %u, %u\n", phi_rng.begin, phi_rng.end, eta_rng.begin, eta_rng.end);
    for (auto i_phi = phi_rng.begin; i_phi != phi_rng.end; i_phi = phi.next_N_bin(i_phi))
    {
        for (auto i_eta = eta_rng.begin; i_eta != eta_rng.end; i_eta = eta.next_N_bin(i_eta))
        {
            printf(" at i_phi=%u, i_eta=%u\n", i_phi, i_eta);
            auto cbin = b.get_content(i_phi, i_eta);
            for (auto i = cbin.first; i < cbin.first + cbin.count; ++i) {
                const track &t = tracks[ b.m_ranks[i] ];
                printf("   %3d  %3d  phi=%f  eta=%f\n", i, b.m_ranks[i], t.phi, t.eta);
            }
        }
    }


    b.reset_contents();

    return 0;
}



// buildtestMPlex.cc::runBtpCe_MultiIter(), loop over seed cleaning multiple times to measure time:
/*
    if ( itconf.m_requires_dupclean_tight ) {
      double t0 = dtime();
      TrackVec xxx; int n_comp;
      for (int i=0;i<1000;++i) {
      xxx = seeds;
      n_comp = StdSeq::clean_cms_seedtracks_iter(&xxx, itconf, eoh.m_beam_spot);
      }
      printf("Seedacleena of %d seeds, out_seeds %d, 1000 times, N_comparisons=%d, took %.5fs\n",
             (int)seeds.size(), (int)xxx.size(), n_comp, dtime() - t0);
      seeds = xxx;
    }
*/

// Example clean seeds using binnor.
// Further enhancements possible by moving the iteration into binnor class (+iterator),
// doing pre-selection on m_cons fine m-bins.
// Perf notes: https://gist.github.com/osschar/2dcd2b01e7c15cc25aa6489f3b242ccb
/*
//=========================================================================
// Seed cleaning (multi-iter)
//=========================================================================
int clean_cms_seedtracks_iter(TrackVec *seed_ptr, const IterationConfig& itrcfg, const BeamSpot &bspot)
{
  const float etamax_brl = Config::c_etamax_brl;
  const float dpt_common = Config::c_dpt_common;

  const float dzmax_bh = itrcfg.m_params.c_dzmax_bh;
  const float drmax_bh = itrcfg.m_params.c_drmax_bh;
  const float dzmax_eh = itrcfg.m_params.c_dzmax_eh;
  const float drmax_eh = itrcfg.m_params.c_drmax_eh;
  const float dzmax_bl = itrcfg.m_params.c_dzmax_bl;
  const float drmax_bl = itrcfg.m_params.c_drmax_bl;
  const float dzmax_el = itrcfg.m_params.c_dzmax_el;
  const float drmax_el = itrcfg.m_params.c_drmax_el;

  const float ptmin_hpt  = itrcfg.m_params.c_ptthr_hpt;

  const float dzmax2_inv_bh = 1.f/(dzmax_bh*dzmax_bh);
  const float drmax2_inv_bh = 1.f/(drmax_bh*drmax_bh);
  const float dzmax2_inv_eh = 1.f/(dzmax_eh*dzmax_eh);
  const float drmax2_inv_eh = 1.f/(drmax_eh*drmax_eh);
  const float dzmax2_inv_bl = 1.f/(dzmax_bl*dzmax_bl);
  const float drmax2_inv_bl = 1.f/(drmax_bl*drmax_bl);
  const float dzmax2_inv_el = 1.f/(dzmax_el*dzmax_el);
  const float drmax2_inv_el = 1.f/(drmax_el*drmax_el);

  // Merge hits from overlapping seeds?
  // For now always true, we require extra hits after seed.
  const bool  merge_hits = true; // itrcfg.merge_seed_hits_during_cleaning();

  if (seed_ptr == nullptr) return 0;
  TrackVec &seeds = *seed_ptr;

  const int ns = seeds.size();
  #ifdef DEBUG
   std::cout << "before seed cleaning "<< seeds.size()<<std::endl;
  #endif
  TrackVec cleanSeedTracks;
  cleanSeedTracks.reserve(ns);
  std::vector<bool> writetrack(ns, true);

  const float invR1GeV = 1.f/Config::track1GeVradius;

  std::vector<int>    nHits(ns);
  std::vector<int>    charge(ns);
  std::vector<float>  oldPhi(ns);
  std::vector<float>  pos2(ns);
  std::vector<float>  eta(ns);
  std::vector<float>  ctheta(ns);
  std::vector<float>  invptq(ns);
  std::vector<float>  pt(ns);
  std::vector<float>  x(ns);
  std::vector<float>  y(ns);
  std::vector<float>  z(ns);
  std::vector<float>  d0(ns);
  int i1,i2; //for the sorting

  axis_pow2_u1<float, unsigned short, 16, 8> ax_phi(-Config::PI, Config::PI);
  axis<float, unsigned short, 8, 8>         ax_eta(-2.6, 2.6, 30u);

  binnor<unsigned int, decltype(ax_phi), decltype(ax_eta), 24, 8> b(ax_phi, ax_eta);
  b.begin_registration(ns);

  for(int ts=0; ts<ns; ts++){
    const Track & tk = seeds[ts];
    nHits[ts] = tk.nFoundHits();
    charge[ts] = tk.charge();
    oldPhi[ts] = tk.momPhi();
    pos2[ts] = std::pow(tk.x(), 2) + std::pow(tk.y(), 2);
    eta[ts] = tk.momEta();
    ctheta[ts] = 1.f/std::tan(tk.theta());
    invptq[ts] = tk.charge()*tk.invpT();
    pt[ts] = tk.pT();
    x[ts] = tk.x();
    y[ts] = tk.y();
    z[ts] = tk.z();
    d0[ts] = tk.d0BeamSpot(bspot.x,bspot.y);

    // If one is sure values are *within* axis ranges:
    // b.register_entry(oldPhi[ts], eta[ts]);
    b.register_entry_safe(oldPhi[ts], eta[ts]);
  }

  b.finalize_registration();

  int n_comparisons = 0;

  // for(int ts=0; ts<ns; ts++){
  for(int sorted_ts=0; sorted_ts<ns; sorted_ts++){
    int ts = b.m_ranks[sorted_ts];

    // printf("Checking sorted_ts=%d ts=%d wwrite=%d\n", sorted_ts, ts, (int) writetrack[ts]);
    if (not writetrack[ts]) continue;//FIXME: this speed up prevents transitive masking; check build cost!

    const float oldPhi1 = oldPhi[ts];
    const float pos2_first = pos2[ts];
    const float Eta1 = eta[ts];
    const float Pt1 = pt[ts];
    const float invptq_first = invptq[ts];

    // To study some more details -- need EventOfHits for this
    int  n_ovlp_hits_added = 0;
    // int  n_ovlp_hits_same_module = 0;
    // int  n_ovlp_hits_shared = 0;
    // int  n_ovlp_tracks = 0;

    auto phi_rng = ax_phi.Rrdr_to_N_bins(oldPhi[ts], 0.08);
    auto eta_rng = ax_eta.Rrdr_to_N_bins(eta[ts], .1);
    // printf("sorted_ts=%d ts=%d -- phi bin range: %u, %u; eta %u, %u\n", sorted_ts, ts, phi_rng.begin, phi_rng.end, eta_rng.begin, eta_rng.end);
    for (auto i_phi = phi_rng.begin; i_phi != phi_rng.end; i_phi = ax_phi.next_N_bin(i_phi))
    {
    for (auto i_eta = eta_rng.begin; i_eta != eta_rng.end; i_eta = ax_eta.next_N_bin(i_eta))
    {
    // printf(" at i_phi=%u, i_eta=%u\n", i_phi, i_eta);
    const auto cbin = b.get_content(i_phi, i_eta);
    for (auto i = cbin.first; i < cbin.end(); ++i)
    {
    //#pragma simd // Vectorization via simd had issues with icc
    // for (int tss= ts+1; tss<ns; tss++)
    //for (int sorted_tss= sorted_ts+1; sorted_tss<ns; sorted_tss++)
    // {
      int tss = b.m_ranks[i];
      if (tss <= ts) continue;

      const float Pt2 = pt[tss];

      ////// Always require charge consistency. If different charge is assigned, do not remove seed-track
      if(charge[tss] != charge[ts])
        continue;

      const float thisDPt = std::abs(Pt2-Pt1);
      ////// Require pT consistency between seeds. If dpT is large, do not remove seed-track.
      if( thisDPt > dpt_common*(Pt1) )
        // continue;
        break; // following seeds will only be farther away in pT

      ++n_comparisons;

      const float Eta2 = eta[tss];
      const float deta2 = std::pow(Eta1-Eta2, 2);

      const float oldPhi2 = oldPhi[tss];

      const float pos2_second = pos2[tss];
      const float thisDXYSign05 = pos2_second > pos2_first ? -0.5f : 0.5f;

      const float thisDXY = thisDXYSign05*sqrt( std::pow(x[ts]-x[tss], 2) + std::pow(y[ts]-y[tss], 2) );

      const float invptq_second = invptq[tss];

      const float newPhi1 = oldPhi1-thisDXY*invR1GeV*invptq_first;
      const float newPhi2 = oldPhi2+thisDXY*invR1GeV*invptq_second;

      const float dphi = cdist(std::abs(newPhi1-newPhi2));

      const float dr2 = deta2+dphi*dphi;

      const float thisDZ = z[ts]-z[tss]-thisDXY*(ctheta[ts]+ctheta[tss]);
      const float dz2 = thisDZ*thisDZ;

      ////// Reject tracks within dR-dz elliptical window.
      ////// Adaptive thresholds, based on observation that duplicates are more abundant at large pseudo-rapidity and low track pT
      bool overlapping = false;
      if(std::abs(Eta1)<etamax_brl){
        if(Pt1>ptmin_hpt){if(dz2*dzmax2_inv_bh+dr2*drmax2_inv_bh<1.0f) overlapping=true; }
        else{if(dz2*dzmax2_inv_bl+dr2*drmax2_inv_bl<1.0f) overlapping=true; }
      }
      else {
        if(Pt1>ptmin_hpt){if(dz2*dzmax2_inv_eh+dr2*drmax2_inv_eh<1.0f) overlapping=true; }
        else{if(dz2*dzmax2_inv_el+dr2*drmax2_inv_el<1.0f) overlapping=true; }
      }

      if(overlapping){
        //Mark tss as a duplicate
        i1=ts;
        i2=tss;
        if (d0[tss]>d0[ts])
          writetrack[tss] = false;
        else {
          writetrack[ts] = false;
          i2 = ts;
          i1 = tss;
        }
        // Add hits from tk2 to the seed we are keeping.
        // NOTE: We only have 3 bits in Track::Status for number of seed hits.
        //       There is a check at entry and after adding of a new hit.
        Track &tk = seeds[i1];
        if (merge_hits && tk.nTotalHits() < 15)
        {
          const Track &tk2 = seeds[i2];
          //We are not actually fitting to the extra hits; use chi2 of 0
          float fakeChi2 = 0.0;

          for (int j = 0; j < tk2.nTotalHits(); ++j)
          {
            int hitidx = tk2.getHitIdx(j);
            int hitlyr = tk2.getHitLyr(j);
            if (hitidx >= 0)
            {
              bool unique = true;
              for (int i = 0; i < tk.nTotalHits(); ++i)
              {
                if ((hitidx == tk.getHitIdx(i)) && (hitlyr == tk.getHitLyr(i))) {
                  unique = false;
                  break;
                }
              }
              if (unique) {
                tk.addHitIdx(tk2.getHitIdx(j), tk2.getHitLyr(j), fakeChi2);
                ++n_ovlp_hits_added;
                if (tk.nTotalHits() >= 15)
                  break;
              }
            }
          }
        }
        if (n_ovlp_hits_added > 0) {
           tk.sortHitsByLayer();
           n_ovlp_hits_added = 0;
        }

        if ( ! writetrack[ts]) goto end_ts_loop;
      }
    } //end of inner loop over tss
    }
    }

    if (writetrack[ts])
    {
      cleanSeedTracks.emplace_back(seeds[ts]);
    }
end_ts_loop: ;
  }

  seeds.swap(cleanSeedTracks);

#ifdef DEBUG
  {
    const int ns2 = seeds.size();
    printf("Number of CMS seeds before %d --> after %d cleaning\n", ns, ns2);

    for (int it = 0; it < ns2; it++)
    {
      const Track& ss = seeds[it];
      printf("  %3i q=%+i pT=%7.3f eta=% 7.3f nHits=%i label=% i\n",
             it,ss.charge(),ss.pT(),ss.momEta(),ss.nFoundHits(),ss.label());
    }
  }
#endif

#ifdef DEBUG  
  std::cout << "AFTER seed cleaning "<< seeds.size()<<std::endl;
#endif

  return n_comparisons; // seeds.size();
}

*/
