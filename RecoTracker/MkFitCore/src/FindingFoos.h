#ifndef RecoTracker_MkFitCore_src_FindingFoos_h
#define RecoTracker_MkFitCore_src_FindingFoos_h

#include "Matrix.h"

namespace mkfit {

  class MkBase;
  class PropagationFlags;

#define COMPUTE_CHI2_ARGS                                                                                    \
  const MPlexLS &, const MPlexLV &, const MPlexQI &, const MPlexHS &, const MPlexHV &, MPlexQF &, MPlexLV &, \
      MPlexQI &, const int, const PropagationFlags &, const bool

#define UPDATE_PARAM_ARGS                                                                                         \
  const MPlexLS &, const MPlexLV &, MPlexQI &, const MPlexHS &, const MPlexHV &, MPlexLS &, MPlexLV &, MPlexQI &, \
      const int, const PropagationFlags &, const bool

  class FindingFoos {
  public:
    void (*m_compute_chi2_foo)(COMPUTE_CHI2_ARGS);
    void (*m_update_param_foo)(UPDATE_PARAM_ARGS);
    void (MkBase::*m_propagate_foo)(float, const int, const PropagationFlags &);

    FindingFoos() {}

    FindingFoos(void (*cch2_f)(COMPUTE_CHI2_ARGS),
                void (*updp_f)(UPDATE_PARAM_ARGS),
                void (MkBase::*p_f)(float, const int, const PropagationFlags &))
        : m_compute_chi2_foo(cch2_f), m_update_param_foo(updp_f), m_propagate_foo(p_f) {}

    static const FindingFoos &get_barrel_finding_foos();
    static const FindingFoos &get_endcap_finding_foos();
    static const FindingFoos &get_finding_foos(bool is_barrel);
  };

}  // end namespace mkfit

#endif
