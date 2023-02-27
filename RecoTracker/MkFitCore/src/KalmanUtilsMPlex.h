#ifndef RecoTracker_MkFitCore_src_KalmanUtilsMPlex_h
#define RecoTracker_MkFitCore_src_KalmanUtilsMPlex_h

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "Matrix.h"

namespace mkfit {

  //------------------------------------------------------------------------------

  enum KalmanFilterOperation { KFO_Calculate_Chi2 = 1, KFO_Update_Params = 2, KFO_Local_Cov = 4 };

  //------------------------------------------------------------------------------

  void kalmanUpdate(const MPlexLS& psErr,
                    const MPlexLV& psPar,
                    const MPlexHS& msErr,
                    const MPlexHV& msPar,
                    MPlexLS& outErr,
                    MPlexLV& outPar,
                    const int N_proc);

  void kalmanPropagateAndUpdate(const MPlexLS& psErr,
                                const MPlexLV& psPar,
                                MPlexQI& Chg,
                                const MPlexHS& msErr,
                                const MPlexHV& msPar,
                                MPlexLS& outErr,
                                MPlexLV& outPar,
                                MPlexQI& outFailFlag,
                                const int N_proc,
                                const PropagationFlags propFlags,
                                const bool propToHit);

  void kalmanComputeChi2(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& inChg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         MPlexQF& outChi2,
                         const int N_proc);

  void kalmanPropagateAndComputeChi2(const MPlexLS& psErr,
                                     const MPlexLV& psPar,
                                     const MPlexQI& inChg,
                                     const MPlexHS& msErr,
                                     const MPlexHV& msPar,
                                     MPlexQF& outChi2,
                                     MPlexLV& propPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags propFlags,
                                     const bool propToHit);

  void kalmanOperation(const int kfOp,
                       const MPlexLS& psErr,
                       const MPlexLV& psPar,
                       const MPlexHS& msErr,
                       const MPlexHV& msPar,
                       MPlexLS& outErr,
                       MPlexLV& outPar,
                       MPlexQF& outChi2,
                       const int N_proc);

  //------------------------------------------------------------------------------

  void kalmanUpdateEndcap(const MPlexLS& psErr,
                          const MPlexLV& psPar,
                          const MPlexHS& msErr,
                          const MPlexHV& msPar,
                          MPlexLS& outErr,
                          MPlexLV& outPar,
                          const int N_proc);

  void kalmanPropagateAndUpdateEndcap(const MPlexLS& psErr,
                                      const MPlexLV& psPar,
                                      MPlexQI& Chg,
                                      const MPlexHS& msErr,
                                      const MPlexHV& msPar,
                                      MPlexLS& outErr,
                                      MPlexLV& outPar,
                                      MPlexQI& outFailFlag,
                                      const int N_proc,
                                      const PropagationFlags propFlags,
                                      const bool propToHit);

  void kalmanComputeChi2Endcap(const MPlexLS& psErr,
                               const MPlexLV& psPar,
                               const MPlexQI& inChg,
                               const MPlexHS& msErr,
                               const MPlexHV& msPar,
                               MPlexQF& outChi2,
                               const int N_proc);

  void kalmanPropagateAndComputeChi2Endcap(const MPlexLS& psErr,
                                           const MPlexLV& psPar,
                                           const MPlexQI& inChg,
                                           const MPlexHS& msErr,
                                           const MPlexHV& msPar,
                                           MPlexQF& outChi2,
                                           MPlexLV& propPar,
                                           MPlexQI& outFailFlag,
                                           const int N_proc,
                                           const PropagationFlags propFlags,
                                           const bool propToHit);

  void kalmanOperationEndcap(const int kfOp,
                             const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             MPlexQF& outChi2,
                             const int N_proc);

}  // end namespace mkfit
#endif
