#ifndef RecoTracker_LSTCore_interface_TrackCandidatesSoA_h
#define RecoTracker_LSTCore_interface_TrackCandidatesSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(TrackCandidatesSoALayout,
                      SOA_COLUMN(short, trackCandidateType),                  // 4-T5 5-pT3 7-pT5 8-pLS
                      SOA_COLUMN(unsigned int, directObjectIndices),          // direct indices to each type containers
                      SOA_COLUMN(ArrayUx2, objectIndices),                    // tracklet and  triplet indices
                      SOA_COLUMN(Params_pT5::ArrayU8xLayers, logicalLayers),  //
                      SOA_COLUMN(Params_pT5::ArrayUxHits, hitIndices),        //
                      SOA_COLUMN(int, pixelSeedIndex),                        //
                      SOA_COLUMN(Params_pT5::ArrayU16xLayers, lowerModuleIndices),  //
                      SOA_COLUMN(FPX, centerX),                                     //
                      SOA_COLUMN(FPX, centerY),                                     //
                      SOA_COLUMN(FPX, radius),                                      //
                      SOA_SCALAR(unsigned int, nTrackCandidates),                   //
                      SOA_SCALAR(unsigned int, nTrackCandidatespT3),                //
                      SOA_SCALAR(unsigned int, nTrackCandidatespT5),                //
                      SOA_SCALAR(unsigned int, nTrackCandidatespLS),                //
                      SOA_SCALAR(unsigned int, nTrackCandidatesT5))                 //

  using TrackCandidatesSoA = TrackCandidatesSoALayout<>;
  using TrackCandidates = TrackCandidatesSoA::View;
  using TrackCandidatesConst = TrackCandidatesSoA::ConstView;
}  // namespace lst
#endif
