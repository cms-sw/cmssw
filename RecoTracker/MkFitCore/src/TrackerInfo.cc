#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <cassert>

namespace mkfit {

  void LayerInfo::set_limits(float r1, float r2, float z1, float z2) {
    m_rin = r1;
    m_rout = r2;
    m_zmin = z1;
    m_zmax = z2;
  }

  void LayerInfo::set_r_hole_range(float rh1, float rh2) {
    m_has_r_range_hole = true;
    m_hole_r_min = rh1;
    m_hole_r_max = rh2;
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

  LayerInfo &TrackerInfo::new_barrel_layer() {
    m_barrel.push_back(new_layer(LayerInfo::Barrel));
    return m_layers.back();
  }

  LayerInfo &TrackerInfo::new_ecap_pos_layer() {
    m_ecap_pos.push_back(new_layer(LayerInfo::EndCapPos));
    return m_layers.back();
  }

  LayerInfo &TrackerInfo::new_ecap_neg_layer() {
    m_ecap_neg.push_back(new_layer(LayerInfo::EndCapNeg));
    return m_layers.back();
  }

}  // end namespace mkfit
