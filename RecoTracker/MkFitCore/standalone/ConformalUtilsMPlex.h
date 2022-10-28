#ifndef RecoTracker_MkFitCore_standalone_ConformalUtilsMPlex_h
#define RecoTracker_MkFitCore_standalone_ConformalUtilsMPlex_h

#include "RecoTracker/MkFitCore/src/Matrix.h"

namespace mkfit {

  // write to iC --> next step will be a propagation no matter what
  void conformalFitMPlex(bool fitting,
                         const MPlexQI seedID,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         const MPlexHV& msPar0,
                         const MPlexHV& msPar1,
                         const MPlexHV& msPar2);

}  // end namespace mkfit

#endif
