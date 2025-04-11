#ifndef RecoTracker_LSTCore_interface_LSTInputSoA_h
#define RecoTracker_LSTCore_interface_LSTInputSoA_h

#ifndef LST_STANDALONE
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#endif

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(InputHitsSoALayout,
                      SOA_COLUMN(float, xs),
                      SOA_COLUMN(float, ys),
                      SOA_COLUMN(float, zs),
                      SOA_COLUMN(unsigned int, idxs),
                      SOA_COLUMN(unsigned int, detid)
#ifndef LST_STANDALONE
                          ,
                      SOA_COLUMN(TrackingRecHit const*, hits)
#endif
  )

  GENERATE_SOA_LAYOUT(InputPixelHitsSoALayout,
                      SOA_COLUMN(Params_pLS::ArrayUxHits, hitIndices),
                      SOA_COLUMN(float, deltaPhi))

  GENERATE_SOA_LAYOUT(InputPixelSeedsSoALayout,
                      SOA_COLUMN(unsigned int, seedIdx),
                      SOA_COLUMN(int, charge),
                      SOA_COLUMN(int, superbin),
                      SOA_COLUMN(PixelType, pixelType),
                      SOA_COLUMN(char, isQuad),
                      SOA_COLUMN(float, ptIn),
                      SOA_COLUMN(float, ptErr),
                      SOA_COLUMN(float, px),
                      SOA_COLUMN(float, py),
                      SOA_COLUMN(float, pz),
                      SOA_COLUMN(float, etaErr),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi))

  using InputHitsSoA = InputHitsSoALayout<>;
  using InputPixelHitsSoA = InputPixelHitsSoALayout<>;
  using InputPixelSeedsSoA = InputPixelSeedsSoALayout<>;

  using InputHits = InputHitsSoA::View;
  using InputHitsConst = InputHitsSoA::ConstView;
  using InputPixelHits = InputPixelHitsSoA::View;
  using InputPixelHitsConst = InputPixelHitsSoA::ConstView;
  using InputPixelSeeds = InputPixelSeedsSoA::View;
  using InputPixelSeedsConst = InputPixelSeedsSoA::ConstView;

}  // namespace lst

#endif
