#include "RecoTracker/MkFitCore/src/MkRZLimits.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <algorithm>
#include <cassert>

namespace mkfit {

  void MkRZLimits::setup(const LayerInfo& li, bool is_outward) {
    m_layer_info_1 = &li;
    m_rin = li.rin();
    m_rout = li.rout();
    m_zmin = li.zmin();
    m_zmax = li.zmax();
    m_is_barrel = li.is_barrel();
    m_is_outward = is_outward;
    m_is_double = false;
  }

  void MkRZLimits::setup(const LayerInfo& li1, const LayerInfo& li2, bool is_outward) {
    assert(li1.layer_type() == li2.layer_type() && "Double layers must consist of single layers of the same type.");

    m_layer_info_1 = &li1;

    m_rin = std::min(li1.rin(), li2.rin());
    m_rout = std::max(li1.rout(), li2.rout());
    m_zmin = std::min(li1.zmin(), li2.zmin());
    m_zmax = std::max(li1.zmax(), li2.zmax());

    m_is_barrel = li1.is_barrel();
    m_is_outward = is_outward;
    m_is_double = true;
  }

  void MkRZLimits::reset() { *this = MkRZLimits(); }

}  // namespace mkfit
