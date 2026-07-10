#ifndef RecoTracker_LSTCore_interface_TrackCandidatesSoA_h
#define RecoTracker_LSTCore_interface_TrackCandidatesSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  // Minimal data content needed for running tracking downstream
  GENERATE_SOA_LAYOUT(TrackCandidatesBaseSoALayout,
                      SOA_COLUMN(Params_TC::ArrayUxHits, hitIndices),
                      SOA_COLUMN(unsigned int, pixelSeedIndex),
                      SOA_COLUMN(LSTObjType, trackCandidateType),
                      SOA_SCALAR(unsigned int, nTrackCandidates))

  GENERATE_SOA_LAYOUT(TrackCandidatesExtendedSoALayout,
                      SOA_COLUMN(ArrayUx2, objectIndices),            // tracklet and  triplet indices
                      SOA_COLUMN(unsigned int, directObjectIndices),  // direct indices to each type containers
                      SOA_COLUMN(Params_TC::ArrayU8xLayers, logicalLayers),
                      SOA_COLUMN(Params_TC::ArrayU16xLayers, lowerModuleIndices),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(FPX, centerX),
                      SOA_COLUMN(FPX, centerY),
                      SOA_COLUMN(FPX, radius),
#endif
                      SOA_SCALAR(unsigned int, nTrackCandidatespT3),
                      SOA_SCALAR(unsigned int, nTrackCandidatespT5),
                      SOA_SCALAR(unsigned int, nTrackCandidatespLS),
                      SOA_SCALAR(unsigned int, nTrackCandidatesT5),
                      SOA_SCALAR(unsigned int, nTrackCandidatesT4))

  using TrackCandidatesBaseSoA = TrackCandidatesBaseSoALayout<>;
  using TrackCandidatesExtendedSoA = TrackCandidatesExtendedSoALayout<>;

  using TrackCandidatesBase = TrackCandidatesBaseSoA::View;
  using TrackCandidatesBaseConst = TrackCandidatesBaseSoA::ConstView;
  using TrackCandidatesExtended = TrackCandidatesExtendedSoA::View;
  using TrackCandidatesExtendedConst = TrackCandidatesExtendedSoA::ConstView;
}  // namespace lst
#endif
