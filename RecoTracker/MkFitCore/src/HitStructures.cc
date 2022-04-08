#include "RecoTracker/MkFitCore/interface/HitStructures.h"

#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "Matriplex/Memory.h"
#include "Ice/IceRevisitedRadix.h"

#include "Debug.h"

namespace mkfit {

  LayerOfHits::~LayerOfHits() {
#ifdef COPY_SORTED_HITS
    free_hits();
#endif
    operator delete[](m_hit_ranks);
  }

#ifdef COPY_SORTED_HITS
  void LayerOfHits::alloc_hits(int size) {
    m_hits = (Hit *)Matriplex::aligned_alloc64(sizeof(Hit) * size);
    m_capacity = size;
    for (int ihit = 0; ihit < m_capacity; ihit++) {
      m_hits[ihit] = Hit();
    }
  }

  void LayerOfHits::free_hits() { std::free(m_hits); }
#endif

  void LayerOfHits::setup_bins(float qmin, float qmax, float dq) {
    // Define layer with min/max and number of bins along q.

    if (dq < 0) {
      m_nq = (int)-dq;
      m_qmin = qmin;
      m_qmax = qmax;
    } else {
      float extent = qmax - qmin;
      m_nq = std::ceil(extent / dq);
      float extra = 0.5f * (m_nq * dq - extent);
      m_qmin = qmin - extra;
      m_qmax = qmax + extra;
    }
    m_fq = m_nq / (qmax - qmin);  // used in e.g. qbin = (q_hit - m_qmin) * m_fq;

    m_phi_bin_infos.resize(m_nq);
    m_phi_bin_deads.resize(m_nq);
  }

  void LayerOfHits::setupLayer(const LayerInfo &li) {
    // Note, LayerInfo::q_bin( ==>  > 0 - bin width, < 0 - number of bins

    assert(m_layer_info == nullptr && "setupLayer() already called.");

    m_layer_info = &li;

    m_is_barrel = m_layer_info->is_barrel();

    if (m_is_barrel)
      setup_bins(li.zmin(), li.zmax(), li.q_bin());
    else
      setup_bins(li.rin(), li.rout(), li.q_bin());
  }

  //==============================================================================

  void LayerOfHits::suckInHits(const HitVec &hitv) {
    assert(m_nq > 0 && "setupLayer() was not called.");

    m_n_hits = hitv.size();
    m_ext_hits = &hitv;

#ifdef COPY_SORTED_HITS
    if (m_capacity < m_n_hits) {
      free_hits();
      alloc_hits(m_n_hits);
    }
#endif

    if (Config::usePhiQArrays) {
      m_hit_phis.resize(m_n_hits);
      m_hit_qs.resize(m_n_hits);
      m_hit_infos.resize(m_n_hits);
    }
    m_qphifines.resize(m_n_hits);

    for (int i = 0; i < m_n_hits; ++i) {
      const Hit &h = hitv[i];

      HitInfo hi = {h.phi(), m_is_barrel ? h.z() : h.r()};

      m_qphifines[i] = phiBinFine(hi.phi) + (qBinChecked(hi.q) << 16);

      if (Config::usePhiQArrays) {
        m_hit_infos[i] = hi;
      }
    }

    operator delete[](m_hit_ranks);
    {
      RadixSort sort;
      sort.Sort(&m_qphifines[0], m_n_hits, RADIX_UNSIGNED);
      m_hit_ranks = sort.RelinquishRanks();
    }

    int curr_qphi = -1;
    empty_q_bins(0, m_nq, 0);

    for (int i = 0; i < m_n_hits; ++i) {
      int j = m_hit_ranks[i];

#ifdef COPY_SORTED_HITS
      memcpy(&m_hits[i], &hitv[j], sizeof(Hit));
#endif

      if (Config::usePhiQArrays) {
        m_hit_phis[i] = m_hit_infos[j].phi;
        m_hit_qs[i] = m_hit_infos[j].q;
      }

      // Combined q-phi bin with fine part masked off
      const int jqphi = m_qphifines[j] & m_phi_fine_xmask;

      const int phi_bin = (jqphi & m_phi_mask_fine) >> m_phi_bits_shift;
      const int q_bin = jqphi >> 16;

      // Fill the bin info
      if (jqphi != curr_qphi) {
        m_phi_bin_infos[q_bin][phi_bin] = {i, i};
        curr_qphi = jqphi;
      }

      m_phi_bin_infos[q_bin][phi_bin].second++;
    }
  }

  //==============================================================================

  void LayerOfHits::suckInDeads(const DeadVec &deadv) {
    assert(m_nq > 0 && "setupLayer() was not called.");

    empty_q_bins_dead(0, m_nq);

    for (const auto &d : deadv) {
      int q_bin_1 = qBinChecked(d.q1);
      int q_bin_2 = qBinChecked(d.q2) + 1;
      int phi_bin_1 = phiBin(d.phi1);
      int phi_bin_2 = phiBin(d.phi2) + 1;
      for (int q_bin = q_bin_1; q_bin < q_bin_2; q_bin++) {
        if (phi_bin_1 > phi_bin_2) {
          for (int pb = phi_bin_1; pb < Config::m_nphi; pb++) {
            m_phi_bin_deads[q_bin][pb] = true;
          }
          for (int pb = 0; pb < phi_bin_2; pb++) {
            m_phi_bin_deads[q_bin][pb] = true;
          }
        } else {
          for (int pb = phi_bin_1; pb < phi_bin_2; pb++) {
            m_phi_bin_deads[q_bin][pb] = true;
          }
        }
      }
    }
  }

  void LayerOfHits::beginRegistrationOfHits(const HitVec &hitv) {
    assert(m_nq > 0 && "setupLayer() was not called.");

    m_ext_hits = &hitv;

    m_n_hits = 0;
    m_hit_infos.clear();
    m_qphifines.clear();
    m_ext_idcs.clear();
    m_min_ext_idx = std::numeric_limits<int>::max();
    m_max_ext_idx = std::numeric_limits<int>::min();
  }

  void LayerOfHits::registerHit(int idx) {
    const Hit &h = (*m_ext_hits)[idx];

    m_ext_idcs.push_back(idx);
    m_min_ext_idx = std::min(m_min_ext_idx, idx);
    m_max_ext_idx = std::max(m_max_ext_idx, idx);

    HitInfo hi = {h.phi(), m_is_barrel ? h.z() : h.r()};

    m_qphifines.push_back(phiBinFine(hi.phi) + (qBinChecked(hi.q) << 16));

    if (Config::usePhiQArrays) {
      m_hit_infos.emplace_back(hi);
    }
  }

  void LayerOfHits::endRegistrationOfHits(bool build_original_to_internal_map) {
    m_n_hits = m_ext_idcs.size();
    if (m_n_hits == 0)
      return;

    // radix
    operator delete[](m_hit_ranks);
    {
      RadixSort sort;
      sort.Sort(&m_qphifines[0], m_n_hits, RADIX_UNSIGNED);
      m_hit_ranks = sort.RelinquishRanks();
    }

    // copy q/phi

#ifdef COPY_SORTED_HITS
    if (m_capacity < m_n_hits) {
      free_hits();
      alloc_hits(m_n_hits);
    }
#endif

    if (Config::usePhiQArrays) {
      m_hit_phis.resize(m_n_hits);
      m_hit_qs.resize(m_n_hits);
    }

    int curr_qphi = -1;
    empty_q_bins(0, m_nq, 0);

    for (int i = 0; i < m_n_hits; ++i) {
      int j = m_hit_ranks[i];  // index in intermediate
      int k = m_ext_idcs[j];   // index in external hit_vec

#ifdef COPY_SORTED_HITS
      memcpy(&m_hits[i], &hitv[k], sizeof(Hit));
#endif

      if (Config::usePhiQArrays) {
        m_hit_phis[i] = m_hit_infos[j].phi;
        m_hit_qs[i] = m_hit_infos[j].q;
      }

      // Combined q-phi bin with fine part masked off
      const int jqphi = m_qphifines[j] & m_phi_fine_xmask;

      const int phi_bin = (jqphi & m_phi_mask_fine) >> m_phi_bits_shift;
      const int q_bin = jqphi >> 16;

      // Fill the bin info
      if (jqphi != curr_qphi) {
        m_phi_bin_infos[q_bin][phi_bin] = {i, i};
        curr_qphi = jqphi;
      }

      m_phi_bin_infos[q_bin][phi_bin].second++;

      // m_hit_ranks[i] will never be used again - use it to point to external/original index.
      m_hit_ranks[i] = k;
    }

    if (build_original_to_internal_map) {
      if (m_max_ext_idx - m_min_ext_idx + 1 > 8 * m_n_hits) {
        // If this happens we might:
        // a) Use external indices for everything. -- *** We are now. ***
        // b) Build these maps for seeding layers only.
        // c) Have a flag in hit-on-track that tells us if the hit index has been remapped,
        //    essentially, if it is a seed hit. This might be smart anyway.
        //    One could use index < -256 or something similar.

        printf(
            "LayerOfHits::endRegistrationOfHits() original_to_internal index map vector is largish: m_n_hits=%d, "
            "map_vector_size=%d\n",
            m_n_hits,
            m_max_ext_idx - m_min_ext_idx + 1);
      }

      m_ext_idcs.resize(m_max_ext_idx - m_min_ext_idx + 1);
      for (int i = 0; i < m_n_hits; ++i) {
        m_ext_idcs[m_hit_ranks[i] - m_min_ext_idx] = i;
      }
    }

    // We can release m_hit_infos and m_qphifines -- and realloc on next BeginInput.
    // m_qphifines could still be used as pre-selection in selectHitIndices().
  }

  //==============================================================================

  /*
  // Example code for looping over a given (q, phi) 2D range.
  // A significantly more complex implementation of this can be found in MkFinder::selectHitIndices().
  void LayerOfHits::selectHitIndices(float q, float phi, float dq, float dphi, std::vector<int>& idcs, bool isForSeeding, bool dump)
  {
    // Sanitizes q, dq and dphi. phi is expected to be in -pi, pi.

    // Make sure how phi bins work beyond -pi, +pi.
    // for (float p = -8; p <= 8; p += 0.05)
    // {
    //   int pb = phiBin(p);
    //   printf("%5.2f %4d %4d\n", p, pb, pb & m_phi_mask);
    // }

    if ( ! isForSeeding) // seeding has set cuts for dq and dphi
    {
      // XXXX MT: min search windows not enforced here.
      dq   = std::min(std::abs(dq),   max_dq());
      dphi = std::min(std::abs(dphi), max_dphi());
    }

    int qb1 = qBinChecked(q - dq);
    int qb2 = qBinChecked(q + dq) + 1;
    int pb1 = phiBin(phi - dphi);
    int pb2 = phiBin(phi + dphi) + 1;

    // int extra = 2;
    // qb1 -= 2; if (qb < 0) qb = 0;
    // qb2 += 2; if (qb >= m_nq) qb = m_nq;

    if (dump)
      printf("LayerOfHits::SelectHitIndices %6.3f %6.3f %6.4f %7.5f %3d %3d %4d %4d\n",
            q, phi, dq, dphi, qb1, qb2, pb1, pb2);

    // This should be input argument, well ... it will be Matriplex op, or sth. // KPM -- it is now! used for seeding
    for (int qi = qb1; qi < qb2; ++qi)
    {
      for (int pi = pb1; pi < pb2; ++pi)
      {
        int pb = pi & m_phi_mask;

        for (uint16_t hi = m_phi_bin_infos[qi][pb].first; hi < m_phi_bin_infos[qi][pb].second; ++hi)
        {
          // Here could enforce some furhter selection on hits
    if (Config::usePhiQArrays)
    {
      float ddq   = std::abs(q   - m_hit_qs[hi]);
      float ddphi = std::abs(phi - m_hit_phis[hi]);
      if (ddphi > Const::PI) ddphi = Const::TwoPI - ddphi;

      if (dump)
        printf("     SHI %3d %4d %4d %5d  %6.3f %6.3f %6.4f %7.5f   %s\n",
        qi, pi, pb, hi,
        m_hit_qs[hi], m_hit_phis[hi], ddq, ddphi,
        (ddq < dq && ddphi < dphi) ? "PASS" : "FAIL");

      if (ddq < dq && ddphi < dphi)
      {
        idcs.push_back(hi);
      }
    }
    else // do not use phi-q arrays
    {
      idcs.push_back(hi);
    }
        }
      }
    }
  }
  */

  void LayerOfHits::printBins() {
    for (int qb = 0; qb < m_nq; ++qb) {
      printf("%c bin %d\n", is_barrel() ? 'Z' : 'R', qb);
      for (int pb = 0; pb < Config::m_nphi; ++pb) {
        if (pb % 8 == 0)
          printf(" Phi %4d: ", pb);
        printf("%5d,%4d   %s",
               m_phi_bin_infos[qb][pb].first,
               m_phi_bin_infos[qb][pb].second,
               ((pb + 1) % 8 == 0) ? "\n" : "");
      }
    }
  }

  //==============================================================================
  // EventOfHits
  //==============================================================================

  EventOfHits::EventOfHits(const TrackerInfo &trk_inf)
      : m_layers_of_hits(trk_inf.n_layers()), m_n_layers(trk_inf.n_layers()) {
    for (int ii = 0; ii < trk_inf.n_layers(); ++ii) {
      const LayerInfo &li = trk_inf.layer(ii);
      m_layers_of_hits[li.layer_id()].setupLayer(li);
    }
  }

}  // end namespace mkfit
