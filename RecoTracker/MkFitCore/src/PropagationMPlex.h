#ifndef RecoTracker_MkFitCore_src_PropagationMPlex_h
#define RecoTracker_MkFitCore_src_PropagationMPlex_h

#include "Matrix.h"

namespace mkfit {

  class PropagationFlags;

  inline void squashPhiMPlex(MPlexLV& par, const int N_proc) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      if (par(n, 4, 0) >= Const::PI)
        par(n, 4, 0) -= Const::TwoPI;
      if (par(n, 4, 0) < -Const::PI)
        par(n, 4, 0) += Const::TwoPI;
    }
  }

  inline void squashPhiMPlexGeneral(MPlexLV& par, const int N_proc) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      par(n, 4, 0) -= std::floor(0.5f * Const::InvPI * (par(n, 4, 0) + Const::PI)) * Const::TwoPI;
    }
  }

  void propagateLineToRMPlex(const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             const int N_proc);

  void propagateHelixToRMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msRad,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags &pflags,
                              const MPlexQI* noMatEffPtr = nullptr);

  void helixAtRFromIterativeCCSFullJac(const MPlexLV& inPar,
                                       const MPlexQI& inChg,
                                       const MPlexQF& msRad,
                                       MPlexLV& outPar,
                                       MPlexLL& errorProp,
                                       MPlexQI& outFailFlag,
                                       const int N_proc);

  void helixAtRFromIterativeCCS(const MPlexLV& inPar,
                                const MPlexQI& inChg,
                                const MPlexQF& msRad,
                                MPlexLV& outPar,
                                MPlexLL& errorProp,
                                MPlexQI& outFailFlag,
                                const int N_proc,
                                const PropagationFlags &pflags);

  void propagateHelixToZMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msZ,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags &pflags,
                              const MPlexQI* noMatEffPtr = nullptr);

  void helixAtZ(const MPlexLV& inPar,
                const MPlexQI& inChg,
                const MPlexQF& msZ,
                MPlexLV& outPar,
                MPlexLL& errorProp,
                MPlexQI& outFailFlag,
                const int N_proc,
                const PropagationFlags &pflags);

  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            const int N_proc,
                            const bool isBarrel);

}  // end namespace mkfit
#endif
