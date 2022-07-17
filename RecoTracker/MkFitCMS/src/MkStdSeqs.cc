#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

#include "RecoTracker/MkFitCore/interface/binnor.h"

#include "oneapi/tbb/parallel_for.h"

namespace mkfit {

  namespace StdSeq {

    //=========================================================================
    // Hit processing
    //=========================================================================

    void loadDeads(EventOfHits &eoh, const std::vector<DeadVec> &deadvectors) {
      for (size_t il = 0; il < deadvectors.size(); il++) {
        eoh.suckInDeads(int(il), deadvectors[il]);
      }
    }

    // Loading hits in CMSSW from two "large multi-layer vectors".
    // orig_hitvectors[0] - pixels,
    // orig_hitvectors[1] - strips.

    void cmssw_LoadHits_Begin(EventOfHits &eoh, const std::vector<const HitVec *> &orig_hitvectors) {
      eoh.reset();
      for (int i = 0; i < eoh.nLayers(); ++i) {
        auto &&l = eoh[i];
        l.beginRegistrationOfHits(*orig_hitvectors[l.is_pixel() ? 0 : 1]);
      }
    }

    // Loop with LayerOfHits::registerHit(int idx) - it takes Hit out of original HitVec to
    // extract phi, r/z, and calculate qphifines
    //
    // Something like what is done in MkFitInputConverter::convertHits
    //
    // Problem is I don't know layers for each large-vector;
    // Also, layer is calculated for each detset when looping over the HitCollection

    void cmssw_LoadHits_End(EventOfHits &eoh) {
      for (int i = 0; i < eoh.nLayers(); ++i) {
        auto &&l = eoh[i];
        l.endRegistrationOfHits(false);
      }
    }

    //=========================================================================
    // Hit-index mapping / remapping
    //=========================================================================

    void cmssw_Map_TrackHitIndices(const EventOfHits &eoh, TrackVec &seeds) {
      for (auto &&track : seeds) {
        for (int i = 0; i < track.nTotalHits(); ++i) {
          const int hitidx = track.getHitIdx(i);
          const int hitlyr = track.getHitLyr(i);
          if (hitidx >= 0) {
            const auto &loh = eoh[hitlyr];
            track.setHitIdx(i, loh.getHitIndexFromOriginal(hitidx));
          }
        }
      }
    }

    void cmssw_ReMap_TrackHitIndices(const EventOfHits &eoh, TrackVec &out_tracks) {
      for (auto &&track : out_tracks) {
        for (int i = 0; i < track.nTotalHits(); ++i) {
          const int hitidx = track.getHitIdx(i);
          const int hitlyr = track.getHitLyr(i);
          if (hitidx >= 0) {
            const auto &loh = eoh[hitlyr];
            track.setHitIdx(i, loh.getOriginalHitIndex(hitidx));
          }
        }
      }
    }

    //=========================================================================
    // Seed cleaning (multi-iter)
    //=========================================================================
    int clean_cms_seedtracks_iter(TrackVec *seed_ptr, const IterationConfig &itrcfg, const BeamSpot &bspot) {
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

      const float ptmin_hpt = itrcfg.m_params.c_ptthr_hpt;

      const float dzmax2_inv_bh = 1.f / (dzmax_bh * dzmax_bh);
      const float drmax2_inv_bh = 1.f / (drmax_bh * drmax_bh);
      const float dzmax2_inv_eh = 1.f / (dzmax_eh * dzmax_eh);
      const float drmax2_inv_eh = 1.f / (drmax_eh * drmax_eh);
      const float dzmax2_inv_bl = 1.f / (dzmax_bl * dzmax_bl);
      const float drmax2_inv_bl = 1.f / (drmax_bl * drmax_bl);
      const float dzmax2_inv_el = 1.f / (dzmax_el * dzmax_el);
      const float drmax2_inv_el = 1.f / (drmax_el * drmax_el);

      // Merge hits from overlapping seeds?
      // For now always true, we require extra hits after seed.
      const bool merge_hits = true;  // itrcfg.merge_seed_hits_during_cleaning();

      if (seed_ptr == nullptr)
        return 0;
      TrackVec &seeds = *seed_ptr;

      const int ns = seeds.size();
#ifdef DEBUG
      std::cout << "before seed cleaning " << seeds.size() << std::endl;
#endif
      TrackVec cleanSeedTracks;
      cleanSeedTracks.reserve(ns);
      std::vector<bool> writetrack(ns, true);

      const float invR1GeV = 1.f / Config::track1GeVradius;

      std::vector<int> nHits(ns);
      std::vector<int> charge(ns);
      std::vector<float> oldPhi(ns);
      std::vector<float> pos2(ns);
      std::vector<float> eta(ns);
      std::vector<float> ctheta(ns);
      std::vector<float> invptq(ns);
      std::vector<float> pt(ns);
      std::vector<float> x(ns);
      std::vector<float> y(ns);
      std::vector<float> z(ns);
      std::vector<float> d0(ns);
      int i1, i2;  //for the sorting

      axis_pow2_u1<float, unsigned short, 16, 8> ax_phi(-Const::PI, Const::PI);
      axis<float, unsigned short, 8, 8> ax_eta(-3.0, 3.0, 30u);
      binnor<unsigned int, decltype(ax_phi), decltype(ax_eta), 24, 8> phi_eta_binnor(ax_phi, ax_eta);

      phi_eta_binnor.begin_registration(ns);

      for (int ts = 0; ts < ns; ts++) {
        const Track &tk = seeds[ts];
        nHits[ts] = tk.nFoundHits();
        charge[ts] = tk.charge();
        oldPhi[ts] = tk.momPhi();
        pos2[ts] = std::pow(tk.x(), 2) + std::pow(tk.y(), 2);
        eta[ts] = tk.momEta();
        ctheta[ts] = 1.f / std::tan(tk.theta());
        invptq[ts] = tk.charge() * tk.invpT();
        pt[ts] = tk.pT();
        x[ts] = tk.x();
        y[ts] = tk.y();
        z[ts] = tk.z();
        d0[ts] = tk.d0BeamSpot(bspot.x, bspot.y);

        phi_eta_binnor.register_entry_safe(oldPhi[ts], eta[ts]);
        // If one is sure values are *within* axis ranges: b.register_entry(oldPhi[ts], eta[ts]);
      }

      phi_eta_binnor.finalize_registration();

      for (int sorted_ts = 0; sorted_ts < ns; sorted_ts++) {
        int ts = phi_eta_binnor.m_ranks[sorted_ts];

        if (not writetrack[ts])
          continue;  // Note: this speed up prevents transitive masking (possibly marginal gain).

        const float oldPhi1 = oldPhi[ts];
        const float pos2_first = pos2[ts];
        const float eta1 = eta[ts];
        const float pt1 = pt[ts];
        const float invptq_first = invptq[ts];

        // To study some more details -- need EventOfHits for this
        int n_ovlp_hits_added = 0;

        auto phi_rng = ax_phi.from_R_rdr_to_N_bins(oldPhi[ts], 0.08f);
        auto eta_rng = ax_eta.from_R_rdr_to_N_bins(eta[ts], .1f);

        for (auto i_phi = phi_rng.begin; i_phi != phi_rng.end; i_phi = ax_phi.next_N_bin(i_phi)) {
          for (auto i_eta = eta_rng.begin; i_eta != eta_rng.end; i_eta = ax_eta.next_N_bin(i_eta)) {
            const auto cbin = phi_eta_binnor.get_content(i_phi, i_eta);
            for (auto i = cbin.first; i < cbin.end(); ++i) {
              int tss = phi_eta_binnor.m_ranks[i];

              if (not writetrack[ts])
                continue;
              if (not writetrack[tss])
                continue;
              if (tss == ts)
                continue;

              const float pt2 = pt[tss];

              // Always require charge consistency. If different charge is assigned, do not remove seed-track
              if (charge[tss] != charge[ts])
                continue;

              const float thisDPt = std::abs(pt2 - pt1);
              // Require pT consistency between seeds. If dpT is large, do not remove seed-track.
              if (thisDPt > dpt_common * pt1)
                continue;

              const float eta2 = eta[tss];
              const float deta2 = std::pow(eta1 - eta2, 2);

              const float oldPhi2 = oldPhi[tss];

              const float pos2_second = pos2[tss];
              const float thisDXYSign05 = pos2_second > pos2_first ? -0.5f : 0.5f;

              const float thisDXY = thisDXYSign05 * sqrt(std::pow(x[ts] - x[tss], 2) + std::pow(y[ts] - y[tss], 2));

              const float invptq_second = invptq[tss];

              const float newPhi1 = oldPhi1 - thisDXY * invR1GeV * invptq_first;
              const float newPhi2 = oldPhi2 + thisDXY * invR1GeV * invptq_second;

              const float dphi = cdist(std::abs(newPhi1 - newPhi2));

              const float dr2 = deta2 + dphi * dphi;

              const float thisDZ = z[ts] - z[tss] - thisDXY * (ctheta[ts] + ctheta[tss]);
              const float dz2 = thisDZ * thisDZ;

              // Reject tracks within dR-dz elliptical window.
              // Adaptive thresholds, based on observation that duplicates are more abundant at large pseudo-rapidity and low track pT
              bool overlapping = false;
              if (std::abs(eta1) < etamax_brl) {
                if (pt1 > ptmin_hpt) {
                  if (dz2 * dzmax2_inv_bh + dr2 * drmax2_inv_bh < 1.0f)
                    overlapping = true;
                } else {
                  if (dz2 * dzmax2_inv_bl + dr2 * drmax2_inv_bl < 1.0f)
                    overlapping = true;
                }
              } else {
                if (pt1 > ptmin_hpt) {
                  if (dz2 * dzmax2_inv_eh + dr2 * drmax2_inv_eh < 1.0f)
                    overlapping = true;
                } else {
                  if (dz2 * dzmax2_inv_el + dr2 * drmax2_inv_el < 1.0f)
                    overlapping = true;
                }
              }

              if (overlapping) {
                //Mark tss as a duplicate
                i1 = ts;
                i2 = tss;
                if (d0[tss] > d0[ts])
                  writetrack[tss] = false;
                else {
                  writetrack[ts] = false;
                  i2 = ts;
                  i1 = tss;
                }
                // Add hits from tk2 to the seed we are keeping.
                // NOTE: We have a limit in Track::Status for the number of seed hits.
                //       There is a check at entry and after adding of a new hit.
                Track &tk = seeds[i1];
                if (merge_hits && tk.nTotalHits() < Track::Status::kMaxSeedHits) {
                  const Track &tk2 = seeds[i2];
                  //We are not actually fitting to the extra hits; use chi2 of 0
                  float fakeChi2 = 0.0;

                  for (int j = 0; j < tk2.nTotalHits(); ++j) {
                    int hitidx = tk2.getHitIdx(j);
                    int hitlyr = tk2.getHitLyr(j);
                    if (hitidx >= 0) {
                      bool unique = true;
                      for (int i = 0; i < tk.nTotalHits(); ++i) {
                        if ((hitidx == tk.getHitIdx(i)) && (hitlyr == tk.getHitLyr(i))) {
                          unique = false;
                          break;
                        }
                      }
                      if (unique) {
                        tk.addHitIdx(tk2.getHitIdx(j), tk2.getHitLyr(j), fakeChi2);
                        ++n_ovlp_hits_added;
                        if (tk.nTotalHits() >= Track::Status::kMaxSeedHits)
                          break;
                      }
                    }
                  }
                }
                if (n_ovlp_hits_added > 0)
                  tk.sortHitsByLayer();
              }
            }  //end of inner loop over tss
          }    //eta bin
        }      //phi bin

        if (writetrack[ts]) {
          cleanSeedTracks.emplace_back(seeds[ts]);
        }
      }

      seeds.swap(cleanSeedTracks);

#ifdef DEBUG
      {
        const int ns2 = seeds.size();
        printf("Number of CMS seeds before %d --> after %d cleaning\n", ns, ns2);

        for (int it = 0; it < ns2; it++) {
          const Track &ss = seeds[it];
          printf("  %3i q=%+i pT=%7.3f eta=% 7.3f nHits=%i label=% i\n",
                 it,
                 ss.charge(),
                 ss.pT(),
                 ss.momEta(),
                 ss.nFoundHits(),
                 ss.label());
        }
      }
#endif

#ifdef DEBUG
      std::cout << "AFTER seed cleaning " << seeds.size() << std::endl;
#endif

      return seeds.size();
    }

    //=========================================================================
    // Duplicate cleaning
    //=========================================================================

    void find_duplicates(TrackVec &tracks) {
      const auto ntracks = tracks.size();
      float eta1, phi1, pt1, deta, dphi, dr2;

      if (ntracks == 0) {
        return;
      }
      for (auto itrack = 0U; itrack < ntracks - 1; itrack++) {
        auto &track = tracks[itrack];
        using Algo = TrackBase::TrackAlgorithm;
        auto const algo = track.algorithm();
        if (algo == Algo::pixelLessStep || algo == Algo::tobTecStep)
          continue;
        eta1 = track.momEta();
        phi1 = track.momPhi();
        pt1 = track.pT();
        for (auto jtrack = itrack + 1; jtrack < ntracks; jtrack++) {
          auto &track2 = tracks[jtrack];
          if (track.label() == track2.label())
            continue;
          if (track.algoint() != track2.algoint())
            continue;

          deta = std::abs(track2.momEta() - eta1);
          if (deta > Config::maxdEta)
            continue;

          dphi = std::abs(squashPhiMinimal(phi1 - track2.momPhi()));
          if (dphi > Config::maxdPhi)
            continue;

          float maxdR = Config::maxdR;
          float maxdRSquared = maxdR * maxdR;
          if (std::abs(eta1) > 2.5f)
            maxdRSquared *= 16.0f;
          else if (std::abs(eta1) > 1.44f)
            maxdRSquared *= 9.0f;
          dr2 = dphi * dphi + deta * deta;
          if (dr2 < maxdRSquared) {
            //Keep track with best score
            if (track.score() > track2.score())
              track2.setDuplicateValue(true);
            else
              track.setDuplicateValue(true);
            continue;
          } else {
            if (pt1 == 0)
              continue;
            if (track2.pT() == 0)
              continue;

            if (std::abs((1 / track2.pT()) - (1 / pt1)) < Config::maxdPt) {
              if (Config::useHitsForDuplicates) {
                float numHitsShared = 0;
                for (int ihit2 = 0; ihit2 < track2.nTotalHits(); ihit2++) {
                  const int hitidx2 = track2.getHitIdx(ihit2);
                  const int hitlyr2 = track2.getHitLyr(ihit2);
                  if (hitidx2 >= 0) {
                    auto const it = std::find_if(track.beginHitsOnTrack(),
                                                 track.endHitsOnTrack(),
                                                 [&hitidx2, &hitlyr2](const HitOnTrack &element) {
                                                   return (element.index == hitidx2 && element.layer == hitlyr2);
                                                 });
                    if (it != track.endHitsOnTrack())
                      numHitsShared++;
                  }
                }

                float fracHitsShared = numHitsShared / std::min(track.nFoundHits(), track2.nFoundHits());
                //Only remove one of the tracks if they share at least X% of the hits (denominator is the shorter track)
                if (fracHitsShared < Config::minFracHitsShared)
                  continue;
              }
              //Keep track with best score
              if (track.score() > track2.score())
                track2.setDuplicateValue(true);
              else
                track.setDuplicateValue(true);
            }  //end of if dPt
          }    //end of else
        }      //end of loop over track2
      }        //end of loop over track1
    }

    void remove_duplicates(TrackVec &tracks) {
      tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](auto track) { return track.getDuplicateValue(); }),
                   tracks.end());
    }

    //=========================================================================
    // SHARED HITS DUPLICATE CLEANING
    //=========================================================================

    void find_duplicates_sharedhits(TrackVec &tracks, const float fraction) {
      const auto ntracks = tracks.size();

      std::vector<float> ctheta(ntracks);
      std::multimap<int, int> hitMap;

      for (auto itrack = 0U; itrack < ntracks; itrack++) {
        auto &trk = tracks[itrack];
        ctheta[itrack] = 1.f / std::tan(trk.theta());
        for (int i = 0; i < trk.nTotalHits(); ++i) {
          if (trk.getHitIdx(i) < 0)
            continue;
          int a = trk.getHitLyr(i);
          int b = trk.getHitIdx(i);
          hitMap.insert(std::make_pair(b * 1000 + a, i > 0 ? itrack : -itrack));  //negative for first hit in trk
        }
      }

      for (auto itrack = 0U; itrack < ntracks; itrack++) {
        auto &trk = tracks[itrack];
        auto phi1 = trk.momPhi();
        auto ctheta1 = ctheta[itrack];

        std::map<int, int> sharingMap;
        for (int i = 0; i < trk.nTotalHits(); ++i) {
          if (trk.getHitIdx(i) < 0)
            continue;
          int a = trk.getHitLyr(i);
          int b = trk.getHitIdx(i);
          auto range = hitMap.equal_range(b * 1000 + a);
          for (auto it = range.first; it != range.second; ++it) {
            if (std::abs(it->second) >= (int)itrack)
              continue;  // don't check your own hits (==) nor sym. checks (>)
            if (i == 0 && it->second < 0)
              continue;  // shared first - first is not counted
            sharingMap[std::abs(it->second)]++;
          }
        }

        for (const auto &elem : sharingMap) {
          auto &track2 = tracks[elem.first];

          // broad dctheta-dphi compatibility checks; keep mostly to preserve consistency with old results
          auto dctheta = std::abs(ctheta[elem.first] - ctheta1);
          if (dctheta > 1.)
            continue;

          auto dphi = std::abs(squashPhiMinimal(phi1 - track2.momPhi()));
          if (dphi > 1.)
            continue;

          if (elem.second >= std::min(trk.nFoundHits(), track2.nFoundHits()) * fraction) {
            if (trk.score() > track2.score())
              track2.setDuplicateValue(true);
            else
              trk.setDuplicateValue(true);
          }
        }  // end sharing hits loop
      }    // end trk loop

      tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](auto track) { return track.getDuplicateValue(); }),
                   tracks.end());
    }

    void find_duplicates_sharedhits_pixelseed(TrackVec &tracks,
                                              const float fraction,
                                              const float drth_central,
                                              const float drth_obarrel,
                                              const float drth_forward) {
      const auto ntracks = tracks.size();

      std::vector<float> ctheta(ntracks);
      for (auto itrack = 0U; itrack < ntracks; itrack++) {
        auto &trk = tracks[itrack];
        ctheta[itrack] = 1.f / std::tan(trk.theta());
      }

      float phi1, invpt1, dctheta, ctheta1, dphi, dr2;
      for (auto itrack = 0U; itrack < ntracks; itrack++) {
        auto &trk = tracks[itrack];
        phi1 = trk.momPhi();
        invpt1 = trk.invpT();
        ctheta1 = ctheta[itrack];
        for (auto jtrack = itrack + 1; jtrack < ntracks; jtrack++) {
          auto &track2 = tracks[jtrack];
          if (trk.label() == track2.label())
            continue;

          dctheta = std::abs(ctheta[jtrack] - ctheta1);

          if (dctheta > Config::maxdcth)
            continue;

          dphi = std::abs(squashPhiMinimal(phi1 - track2.momPhi()));

          if (dphi > Config::maxdphi)
            continue;

          float maxdRSquared = drth_central * drth_central;
          if (std::abs(ctheta1) > Config::maxcth_fw)
            maxdRSquared = drth_forward * drth_forward;
          else if (std::abs(ctheta1) > Config::maxcth_ob)
            maxdRSquared = drth_obarrel * drth_obarrel;
          dr2 = dphi * dphi + dctheta * dctheta;
          if (dr2 < maxdRSquared) {
            //Keep track with best score
            if (trk.score() > track2.score())
              track2.setDuplicateValue(true);
            else
              trk.setDuplicateValue(true);
            continue;
          }

          if (std::abs(track2.invpT() - invpt1) > Config::maxd1pt)
            continue;

          auto sharedCount = 0;
          auto sharedFirst = 0;
          const auto minFoundHits = std::min(trk.nFoundHits(), track2.nFoundHits());

          for (int i = 0; i < trk.nTotalHits(); ++i) {
            if (trk.getHitIdx(i) < 0)
              continue;
            const int a = trk.getHitLyr(i);
            const int b = trk.getHitIdx(i);
            for (int j = 0; j < track2.nTotalHits(); ++j) {
              if (track2.getHitIdx(j) < 0)
                continue;
              const int c = track2.getHitLyr(j);
              const int d = track2.getHitIdx(j);

              //this is to count once shared matched hits (may be done more properly...)
              if (a == c && b == d)
                sharedCount += 1;
              if (j == 0 && i == 0 && a == c && b == d)
                sharedFirst += 1;

              if ((sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction))
                continue;
            }
            if ((sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction))
              continue;
          }

          //selection here - 11percent fraction of shared hits to label a duplicate
          if ((sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction)) {
            if (trk.score() > track2.score())
              track2.setDuplicateValue(true);
            else
              trk.setDuplicateValue(true);
          }
        }
      }  //end loop one over tracks

      //removal here
      tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](auto track) { return track.getDuplicateValue(); }),
                   tracks.end());
    }

    //=========================================================================
    //
    //=========================================================================

    void find_and_remove_duplicates(TrackVec &tracks, const IterationConfig &itconf) {
#ifdef DEBUG
      std::cout << " find_and_remove_duplicates: input track size " << tracks.size() << std::endl;
#endif
      if (itconf.m_requires_quality_filter && !(itconf.m_requires_dupclean_tight)) {
        find_duplicates_sharedhits(tracks, itconf.m_params.fracSharedHits);
      } else if (itconf.m_requires_dupclean_tight) {
        find_duplicates_sharedhits_pixelseed(tracks,
                                             itconf.m_params.fracSharedHits,
                                             itconf.m_params.drth_central,
                                             itconf.m_params.drth_obarrel,
                                             itconf.m_params.drth_forward);
      } else {
        find_duplicates(tracks);
        remove_duplicates(tracks);
      }

#ifdef DEBUG
      std::cout << " find_and_remove_duplicates: output track size " << tracks.size() << std::endl;
      for (auto const &tk : tracks) {
        std::cout << tk.parameters() << std::endl;
      }
#endif
    }

  }  // namespace StdSeq
}  // namespace mkfit
