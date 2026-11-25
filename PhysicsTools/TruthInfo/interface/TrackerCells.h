// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// The cell inside a tracker module, named the same way on the truth side and on the reco
// side. A tracker DetId names a module, not a cell, so two particles crossing one module
// share every DetId they leave there. The digi channel separates them: the digitizer
// records it for each fired cell and it is the same number a cluster carries.
//
// The truth side never decodes the channel, it stores what the sim link holds, so only
// the reco side packs one. The two packings differ and are not interchangeable: the
// inner tracker puts the row in the high bits, the outer tracker puts the column there.
// Pick by cluster type, which is what the reco object carries, rather than by
// subdetector.
//
// Both trackers are keyed. The key removes the unrelated truth particles a module-level
// DetId lets compete for a track. Measured on 20 ttbar events at PU200, tracks above
// 1 GeV, not counting the ancestors and descendants of the best match: the unrelated
// particles sharing at least a tenth of a track are 193 on average keyed by module,
// 10.7 with the inner tracker keyed by cell, and 0.09 with both, when 97.4% of tracks
// have none. The index costs 1.49 MB per event more on disk, 16% of its size.

#ifndef PhysicsTools_TruthInfo_interface_TrackerCells_h
#define PhysicsTools_TruthInfo_interface_TrackerCells_h

#include <algorithm>
#include <cstdint>

#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  // The digi channel of an inner-tracker cell, the packing
  // Phase2TrackerDigitizerAlgorithm writes into the sim links of a pixel module.
  [[nodiscard]] inline uint32_t pixelCell(int row, int column) {
    return static_cast<uint32_t>(PixelDigi::pixelToChannel(row, column));
  }

  // The same for an outer-tracker strip. The digitizer writes this packing whenever the
  // module is not a pixel one, and it is NOT the packing above: the two swap the fields.
  [[nodiscard]] inline uint32_t outerTrackerCell(unsigned int row, unsigned int column) {
    return static_cast<uint32_t>(Phase2TrackerDigi::pixelToChannel(row, column));
  }

  // True for an index whose tracker truth carries hits and no cell. No track matches it.
  [[nodiscard]] inline bool isModuleKeyedTracker(LogicalGraphHitIndex const& index) {
    auto const& hits = index.channel(HitChannel::Tracker).directHits;
    return !hits.empty() &&
           std::none_of(hits.begin(), hits.end(), [](LogicalGraphHitIndex::Hit const& h) { return h.hasCell(); });
  }

}  // namespace truth

#endif
