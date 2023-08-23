#ifndef RecoTracker_MkFitCore_interface_SteeringParams_h
#define RecoTracker_MkFitCore_interface_SteeringParams_h

#include "RecoTracker/MkFitCore/interface/FunctionTypes.h"

#include <vector>
#include <stdexcept>

namespace mkfit {

  //==============================================================================
  // LayerControl
  //==============================================================================

  struct LayerControl {
    int m_layer;

    // Idea only ... need some parallel structure for candidates to make sense (where i can store it).
    // Or have per layer containers where I place track indices to enable. Or something. Sigh.
    // int  m_on_miss_jump_to = -999;
    // int  m_on_hit_jump_to  = -999;

    // Used to have pickup-only / bk-fit only bools etc.
    // Moved to SteeringParams as layer indices where pickup/bkfit/bksrch start/end/start.

    //----------------------------------------------------------------------------

    LayerControl() : m_layer(-1) {}
    LayerControl(int lay) : m_layer(lay) {}
  };

  //==============================================================================
  // SteeringParams
  //==============================================================================

  class SteeringParams {
  public:
    enum IterationType_e { IT_FwdSearch, IT_BkwFit, IT_BkwSearch };

    class iterator {
      friend class SteeringParams;

      const SteeringParams& m_steering_params;
      IterationType_e m_type;
      int m_cur_index = -1;
      int m_end_index = -1;

      iterator(const SteeringParams& sp, IterationType_e t) : m_steering_params(sp), m_type(t) {}

    public:
      const LayerControl& layer_control() const { return m_steering_params.m_layer_plan[m_cur_index]; }
      int layer() const { return layer_control().m_layer; }
      int index() const { return m_cur_index; }
      int region() const { return m_steering_params.m_region; }

      bool is_valid() const { return m_cur_index != -1; }

      const LayerControl& operator->() const { return layer_control(); }

      bool is_pickup_only() const {
        if (m_type == IT_FwdSearch)
          return m_cur_index == m_steering_params.m_fwd_search_pickup;
        else if (m_type == IT_BkwSearch)
          return m_cur_index == m_steering_params.m_bkw_search_pickup;
        else
          throw std::runtime_error("invalid iteration type");
      }

      bool operator++() {
        if (!is_valid())
          return false;
        if (m_type == IT_FwdSearch) {
          if (++m_cur_index == m_end_index)
            m_cur_index = -1;
        } else {
          if (--m_cur_index == m_end_index)
            m_cur_index = -1;
        }
        return is_valid();
      }

      // Functions for debug printouts
      int end_index() const { return m_end_index; }
      int next_layer() const {
        if (m_type == IT_FwdSearch)
          return m_steering_params.m_layer_plan[m_cur_index + 1].m_layer;
        else
          return m_steering_params.m_layer_plan[m_cur_index - 1].m_layer;
      }
      int last_layer() const {
        if (m_type == IT_FwdSearch)
          return m_steering_params.m_layer_plan[m_end_index - 1].m_layer;
        else
          return m_steering_params.m_layer_plan[m_end_index + 1].m_layer;
      }
    };  // class iterator

    std::vector<LayerControl> m_layer_plan;
    track_score_func m_track_scorer;
    std::string m_track_scorer_name;

    int m_region;

    int m_fwd_search_pickup = 0;
    int m_bkw_fit_last = 0;
    int m_bkw_search_pickup = -1;

    //----------------------------------------------------------------------------

    SteeringParams() {}

    void reserve_plan(int n) { m_layer_plan.reserve(n); }

    void append_plan(int layer) { m_layer_plan.emplace_back(LayerControl(layer)); }

    void fill_plan(int first, int last) {
      for (int i = first; i <= last; ++i)
        append_plan(i);
    }

    void set_iterator_limits(int fwd_search_pu, int bkw_fit_last, int bkw_search_pu = -1) {
      m_fwd_search_pickup = fwd_search_pu;
      m_bkw_fit_last = bkw_fit_last;
      m_bkw_search_pickup = bkw_search_pu;
    }

    bool has_bksearch_plan() const { return m_bkw_search_pickup != -1; }

    iterator make_iterator(IterationType_e type) const {
      iterator it(*this, type);

      if (type == IT_FwdSearch) {
        it.m_cur_index = m_fwd_search_pickup;
        it.m_end_index = m_layer_plan.size();
      } else if (type == IT_BkwFit) {
        it.m_cur_index = m_layer_plan.size() - 1;
        it.m_end_index = m_bkw_fit_last - 1;
      } else if (type == IT_BkwSearch) {
        it.m_cur_index = m_bkw_search_pickup;
        it.m_end_index = -1;
      } else
        throw std::invalid_argument("unknown iteration type");

      if (!it.is_valid())
        throw std::runtime_error("invalid iterator constructed");

      return it;
    }
  };

}  // end namespace mkfit

#endif
