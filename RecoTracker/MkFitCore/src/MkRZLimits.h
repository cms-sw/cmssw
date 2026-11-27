#ifndef RecoTracker_MkFitCore_src_MkRZLimits_h
#define RecoTracker_MkFitCore_src_MkRZLimits_h

namespace mkfit {

  class LayerInfo;

  struct MkRZLimits {
    const LayerInfo* m_layer_info_1 = nullptr;
    float m_rin = 0, m_rout = 0, m_zmin = 0, m_zmax = 0;
    bool m_is_barrel = true;
    bool m_is_outward = true;
    bool m_is_double = false;

    void setup(const LayerInfo& li, bool is_outward);
    void setup(const LayerInfo& li1, const LayerInfo& li2, bool is_outward);

    void reset();

    const LayerInfo& layer_info_1() const { return *m_layer_info_1; }

    bool is_outward() const { return m_is_outward; }
    bool is_inward() const { return !m_is_outward; }

    // False if the track is already past the layer in z and moving away from it.
    // Exact for a helix, where z is monotonic along the track; does not test r
    // (the WSR verdict does) and has no margin for the track's z error.
    bool rz_quadrant_check(float z, float pz) const {
      if (is_inward()) {
        pz = -pz;
      }
      if (pz < 0.0f) {
        if (z < m_zmin)
          return false;
      } else {
        if (z > m_zmax)
          return false;
      }
      return true;
    }
  };

}  // namespace mkfit

#endif
