#ifndef RecoTracker_LSTCore_interface_LSTInputSoA_h
#define RecoTracker_LSTCore_interface_LSTInputSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(HitsBaseSoALayout,
                      SOA_COLUMN(float, xs),
                      SOA_COLUMN(float, ys),
                      SOA_COLUMN(float, zs),
                      SOA_COLUMN(unsigned int, detid),
                      SOA_COLUMN(uint16_t, clustsize),
                      SOA_SCALAR(unsigned int, nHitsOT))

  GENERATE_SOA_LAYOUT(PixelSeedsSoALayout,
                      SOA_COLUMN(unsigned int, firstHit),
                      SOA_COLUMN(uint8_t, nHits),
                      SOA_COLUMN(uint8_t, hitDetBits),  // IT 0, OT 1 up to min(nHits, 8)
                      SOA_COLUMN(float, deltaPhi),
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

  // Original index of each hit of the pLS section (hits nHitsOT and up); an OT hit's original index is its own index.
  GENERATE_SOA_LAYOUT(HitsITSoALayout, SOA_COLUMN(unsigned int, idxs))

  GENERATE_SOA_BLOCKS(LSTInputSoALayout,
                      SOA_BLOCK(hits, HitsBaseSoALayout),
                      SOA_BLOCK(pixelSeeds, PixelSeedsSoALayout),
                      SOA_BLOCK(hitsIT, HitsITSoALayout))

  using HitsBaseSoA = HitsBaseSoALayout<>;
  using PixelSeedsSoA = PixelSeedsSoALayout<>;
  using HitsITSoA = HitsITSoALayout<>;
  using LSTInputSoA = LSTInputSoALayout<>;

  using HitsBase = HitsBaseSoA::View;
  using HitsBaseConst = HitsBaseSoA::ConstView;
  using PixelSeeds = PixelSeedsSoA::View;
  using PixelSeedsConst = PixelSeedsSoA::ConstView;
  using HitsIT = HitsITSoA::View;
  using HitsITConst = HitsITSoA::ConstView;
  using LSTInputView = LSTInputSoA::View;
  using LSTInputConstView = LSTInputSoA::ConstView;

  // Original (input collection) index of LST hit ih.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned int hitOrigIdx(HitsBaseConst hits, HitsITConst hitsIT, unsigned int ih) {
    unsigned int const nHitsOT = hits.nHitsOT();
    return ih < nHitsOT ? ih : hitsIT.idxs()[ih - nHitsOT];
  }

}  // namespace lst

#endif
