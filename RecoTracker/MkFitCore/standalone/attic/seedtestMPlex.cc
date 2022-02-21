#include "seedtestMPlex.h"
#include "oneapi/tbb/parallel_for.h"

// #define DEBUG
#include "Debug.h"

namespace mkfit {

  inline void intersectThirdLayer(
      const float a, const float b, const float hit1_x, const float hit1_y, float& lay2_x, float& lay2_y) {
    const float a2 = a * a;
    const float b2 = b * b;
    const float a2b2 = a2 + b2;
    const float lay2rad2 = (Config::fRadialSpacing * Config::fRadialSpacing) * 9.0f;  // average third radius squared
    const float maxCurvR2 = Config::maxCurvR * Config::maxCurvR;

    const float quad =
        std::sqrt(2.0f * maxCurvR2 * (a2b2 + lay2rad2) - (a2b2 - lay2rad2) * (a2b2 - lay2rad2) - maxCurvR2 * maxCurvR2);
    const float pos[2] = {(a2 * a + a * (b2 + lay2rad2 - maxCurvR2) - b * quad) / a2b2,
                          (b2 * b + b * (a2 + lay2rad2 - maxCurvR2) + a * quad) / a2b2};
    const float neg[2] = {(a2 * a + a * (b2 + lay2rad2 - maxCurvR2) + b * quad) / a2b2,
                          (b2 * b + b * (a2 + lay2rad2 - maxCurvR2) - a * quad) / a2b2};

    // since we have two intersection points, arbitrate which one is closer to layer2 hit
    if (getHypot(pos[0] - hit1_x, pos[1] - hit1_y) < getHypot(neg[0] - hit1_x, neg[1] - hit1_y)) {
      lay2_x = pos[0];
      lay2_y = pos[1];
    } else {
      lay2_x = neg[0];
      lay2_y = neg[1];
    }
  }

  void findSeedsByRoadSearch(TripletIdxConVec& seed_idcs,
                             std::vector<LayerOfHits>& evt_lay_hits,
                             int lay1_size,
                             Event*& ev) {
#ifdef DEBUG
    bool debug(false);
#endif

    // MIMI hack: Config::nlayers_per_seed = 4
    // const float seed_z2cut = (Config::nlayers_per_seed * Config::fRadialSpacing) / std::tan(2.0f*std::atan(std::exp(-1.0f*Config::dEtaSeedTrip)));
#ifdef DEBUG
    const float seed_z2cut =
        (4 * Config::fRadialSpacing) / std::tan(2.0f * std::atan(std::exp(-1.0f * Config::dEtaSeedTrip)));
#endif

    // 0 = first layer, 1 = second layer, 2 = third layer
    const LayerOfHits& lay1_hits = evt_lay_hits[1];
    LayerOfHits& lay0_hits = evt_lay_hits[0];
    LayerOfHits& lay2_hits = evt_lay_hits[2];

    tbb::parallel_for(
        tbb::blocked_range<int>(0, lay1_size, std::max(1, Config::numHitsPerTask)),
        [&](const tbb::blocked_range<int>& i) {
          TripletIdxVec temp_thr_seed_idcs;
          for (int ihit1 = i.begin(); ihit1 < i.end(); ++ihit1) {
            const Hit& hit1 = lay1_hits.refHit(ihit1);
            const float hit1_z = hit1.z();

            dprint("ihit1: " << ihit1 << " mcTrackID: " << hit1.mcTrackID(ev->simHitsInfo_) << " phi: " << hit1.phi()
                             << " z: " << hit1.z());
            dprint(" predphi: " << hit1.phi() << "+/-" << Config::lay01angdiff << " predz: " << hit1.z() / 2.0f << "+/-"
                                << Config::seed_z0cut / 2.0f << std::endl);

            std::vector<int> cand_hit0_indices;  // pass by reference
            // MIMI lay0_hits.selectHitIndices(hit1_z/2.0f,hit1.phi(),Config::seed_z0cut/2.0f,Config::lay01angdiff,cand_hit0_indices,true,false);
            // loop over first layer hits
            for (auto&& ihit0 : cand_hit0_indices) {
              const Hit& hit0 = lay0_hits.refHit(ihit0);
              const float hit0_z = hit0.z();
              const float hit0_x = hit0.x();
              const float hit0_y = hit0.y();
              const float hit1_x = hit1.x();
              const float hit1_y = hit1.y();
              const float hit01_r2 = getRad2(hit0_x - hit1_x, hit0_y - hit1_y);

              const float quad = std::sqrt((4.0f * Config::maxCurvR * Config::maxCurvR - hit01_r2) / hit01_r2);

              // center of negative curved track
              const float aneg = 0.5f * ((hit0_x + hit1_x) - (hit0_y - hit1_y) * quad);
              const float bneg = 0.5f * ((hit0_y + hit1_y) + (hit0_x - hit1_x) * quad);

              // negative points of intersection with third layer
              float lay2_negx = 0.0f, lay2_negy = 0.0f;
              intersectThirdLayer(aneg, bneg, hit1_x, hit1_y, lay2_negx, lay2_negy);
#ifdef DEBUG
              const float lay2_negphi = getPhi(lay2_negx, lay2_negy);
#endif

              // center of positive curved track
              const float apos = 0.5f * ((hit0_x + hit1_x) + (hit0_y - hit1_y) * quad);
              const float bpos = 0.5f * ((hit0_y + hit1_y) - (hit0_x - hit1_x) * quad);

              // positive points of intersection with third layer
              float lay2_posx = 0.0f, lay2_posy = 0.0f;
              intersectThirdLayer(apos, bpos, hit1_x, hit1_y, lay2_posx, lay2_posy);
#ifdef DEBUG
              const float lay2_posphi = getPhi(lay2_posx, lay2_posy);
#endif

              std::vector<int> cand_hit2_indices;
              // MIMI lay2_hits.selectHitIndices((2.0f*hit1_z-hit0_z),(lay2_posphi+lay2_negphi)/2.0f,
              // MIMI seed_z2cut,(lay2_posphi-lay2_negphi)/2.0f,
              // MIMI cand_hit2_indices,true,false);

              dprint(" ihit0: " << ihit0 << " mcTrackID: " << hit0.mcTrackID(ev->simHitsInfo_) << " phi: " << hit0.phi()
                                << " z: " << hit0.z());
              dprint("  predphi: " << (lay2_posphi + lay2_negphi) / 2.0f << "+/-" << (lay2_posphi - lay2_negphi) / 2.0f
                                   << " predz: " << 2.0f * hit1_z - hit0_z << "+/-" << seed_z2cut << std::endl);

          // loop over candidate third layer hits
          //temp_thr_seed_idcs.reserve(temp_thr_seed_idcs.size()+cand_hit2_indices.size());
#pragma omp simd
              for (size_t idx = 0; idx < cand_hit2_indices.size(); ++idx) {
                const int ihit2 = cand_hit2_indices[idx];
                const Hit& hit2 = lay2_hits.refHit(ihit2);

                const float lay1_predz = (hit0_z + hit2.z()) / 2.0f;
                // filter by residual of second layer hit
                if (std::abs(lay1_predz - hit1_z) > Config::seed_z1cut)
                  continue;

                const float hit2_x = hit2.x();
                const float hit2_y = hit2.y();

                // now fit a circle, extract pT and d0 from center and radius
                const float mr = (hit1_y - hit0_y) / (hit1_x - hit0_x);
                const float mt = (hit2_y - hit1_y) / (hit2_x - hit1_x);
                const float a = (mr * mt * (hit2_y - hit0_y) + mr * (hit1_x + hit2_x) - mt * (hit0_x + hit1_x)) /
                                (2.0f * (mr - mt));
                const float b = -1.0f * (a - (hit0_x + hit1_x) / 2.0f) / mr + (hit0_y + hit1_y) / 2.0f;
                const float r = getHypot(hit0_x - a, hit0_y - b);

                // filter by d0 cut 5mm, pT cut 0.5 GeV (radius of 0.5 GeV track)
                if ((r < Config::maxCurvR) || (std::abs(getHypot(a, b) - r) > Config::seed_d0cut))
                  continue;

                dprint(" ihit2: " << ihit2 << " mcTrackID: " << hit2.mcTrackID(ev->simHitsInfo_)
                                  << " phi: " << hit2.phi() << " z: " << hit2.z());

                temp_thr_seed_idcs.emplace_back(TripletIdx{{ihit0, ihit1, ihit2}});
              }  // end loop over third layer matches
            }    // end loop over first layer matches
          }      // end chunk of hits for parallel for
          seed_idcs.grow_by(temp_thr_seed_idcs.begin(), temp_thr_seed_idcs.end());
        });  // end parallel for loop over second layer hits
  }

}  // end namespace mkfit
