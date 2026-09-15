// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// The cell inside a tracker module, named the same way on the truth side and on the reco
// side. A tracker DetId names a module, not a cell, so two particles crossing one module
// share every DetId they leave there. The digi channel separates them: the digitizer
// records it for each fired cell and it is the same number a cluster's pixels carry.
//
// Only the inner tracker is keyed this way. Measured on 20 ttbar events at PU200, a
// module shared with the matched branch carries another particle's cluster 8.6% of the
// time in the inner barrel and 6.1% in the inner endcap, against 1.8% and 0.8% in the
// outer tracker, where a cell key would cost entries for almost no gain.

#ifndef PhysicsTools_TruthInfo_interface_TrackerCells_h
#define PhysicsTools_TruthInfo_interface_TrackerCells_h

#include <cstdint>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

namespace truth {

  // Whether the truth of this module is keyed by cell. The two sides must agree, so both
  // the index producer and the reco adapters ask this one function.
  [[nodiscard]] inline bool isCellKeyedSubdetector(uint32_t rawDetId) {
    const int subdet = DetId(rawDetId).subdetId();
    return subdet == PixelSubdetector::PixelBarrel || subdet == PixelSubdetector::PixelEndcap;
  }

  // The digi channel of an inner-tracker cell, the packing Phase2TrackerDigitizerAlgorithm
  // writes into the sim links for a pixel module.
  [[nodiscard]] inline uint32_t pixelCell(int row, int column) {
    return static_cast<uint32_t>(PixelDigi::pixelToChannel(row, column));
  }

}  // namespace truth

#endif
