#ifndef RecoTracker_MkFitCore_src_MatriplexPackers_h
#define RecoTracker_MkFitCore_src_MatriplexPackers_h

#include "Matrix.h"

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

namespace mkfit {

  //==============================================================================
  // MatriplexPackerSlurpIn
  //==============================================================================

  template <typename D>
  class MatriplexPackerSlurpIn {
  protected:
    alignas(64) int m_idx[NN];

    const D* m_base;
    int m_pos;

  public:
    MatriplexPackerSlurpIn(const D& base) : m_base(&base), m_pos(0) {}

    void reset() { m_pos = 0; }

    void addNullInput() { m_idx[m_pos++] = 0; }

    void addInput(const D& item) {
      // Could issue prefetch requests here.

      m_idx[m_pos] = &item - m_base;

      ++m_pos;
    }

    void addInputAt(int pos, const D& item) {
      while (m_pos < pos) {
        // We might not care about initialization / reset to 0.
        // Or we could be building an additional mask (on top of N_proc).
        m_idx[m_pos++] = 0;
      }

      addInput(item);
    }

    template <typename TM>
    void pack(TM& mplex, int base_offset) {
      assert(m_pos > 0 && m_pos <= NN);

#if defined(GATHER_INTRINSICS)
      GATHER_IDX_LOAD(vi, m_idx);
      mplex.slurpIn(m_base + base_offset, vi, D(), m_pos);
#else
      mplex.slurpIn(m_base + base_offset, m_idx, m_pos);
#endif
    }
  };

  //==============================================================================
  // MatriplexErrParPackerSlurpIn
  //==============================================================================

  // T - input class (Track or Hit), D - data type (float)

  template <typename T, typename D>
  class MatriplexErrParPackerSlurpIn : public MatriplexPackerSlurpIn<D> {
    int m_off_param;

  public:
    MatriplexErrParPackerSlurpIn(const T& t)
        : MatriplexPackerSlurpIn<D>(*t.errArray()), m_off_param(t.posArray() - this->m_base) {}

    void addInput(const T& item) {
      // Could issue L1 prefetch requests here.

      this->m_idx[this->m_pos] = item.errArray() - this->m_base;

      ++this->m_pos;
    }

    void addInputAt(int pos, const T& item) {
      while (this->m_pos < pos) {
        // We might not care about initialization / reset to 0.
        // Or we could be building an additional mask (on top of N_proc).
        this->m_idx[this->m_pos++] = 0;
      }

      addInput(item);
    }

    template <typename TMerr, typename TMpar>
    void pack(TMerr& err, TMpar& par) {
      assert(this->m_pos > 0 && this->m_pos <= NN);

#if defined(GATHER_INTRINSICS)
      GATHER_IDX_LOAD(vi, this->m_idx);
      err.slurpIn(this->m_base, vi, D(), this->m_pos);
      par.slurpIn(this->m_base + m_off_param, vi, D(), this->m_pos);
#else
      err.slurpIn(this->m_base, this->m_idx, this->m_pos);
      par.slurpIn(this->m_base + m_off_param, this->m_idx, this->m_pos);
#endif
    }
  };

  //==============================================================================
  // MatriplexTrackPackerPlexify
  //==============================================================================

  template <typename T, typename D>
  class MatriplexTrackPackerPlexify  // : public MatriplexTrackPackerBase
  {
  public:
    MatriplexTrackPackerPlexify(const T& t) {}

    void reset() {}

    void addNullInput() {}

    void addInput(const T& item) {}

    void addInputAt(int pos, const T& item) {}

    template <typename TMerr, typename TMpar>
    void pack(TMerr& err, TMpar& par) {}
  };

  //==============================================================================
  // Packer Selection
  //==============================================================================

  // Optionally ifdef with defines from Makefile.config

  using MatriplexHitPacker = MatriplexErrParPackerSlurpIn<Hit, float>;
  using MatriplexTrackPacker = MatriplexErrParPackerSlurpIn<TrackBase, float>;

  using MatriplexHoTPacker = MatriplexPackerSlurpIn<HitOnTrack>;
}  // namespace mkfit

#endif
