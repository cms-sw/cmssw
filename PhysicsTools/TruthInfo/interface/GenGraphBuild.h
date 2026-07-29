// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// The HepMC record flattened into the arrays a TruthGraph GEN half needs: the
// vertex and particle barcodes, the two edge lists, and the per-barcode pdgId,
// status and packed reco::GenStatusFlags.
//
// Shared by TruthGraphProducer, which builds the graph from an unmixed event, and
// by TruthGraphAccumulator, which builds it per sub-event during mixing. Both must
// produce the same GEN half for the same HepMC record.

#ifndef PhysicsTools_TruthInfo_GenGraphBuild_h
#define PhysicsTools_TruthInfo_GenGraphBuild_h

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace HepMC {
  class GenEvent;
}
namespace HepMC3 {
  class GenEvent;
}

namespace truth {

  // Node keys for the GEN realm. A GenVertex and a GenParticle can carry the same
  // barcode, so the low bit separates them.
  [[nodiscard]] inline int64_t genKeyVertex(int barcode) { return (static_cast<int64_t>(barcode) << 1) | 1LL; }
  [[nodiscard]] inline int64_t genKeyParticle(int barcode) { return static_cast<int64_t>(barcode) << 1; }

  struct GenBuild {
    std::vector<int> vtxBarcodes;
    std::vector<int> partBarcodes;

    // index -> barcode in HepMC iteration order, for diagnostics only.
    // SimTrack::genpartIndex() is a barcode, not an index into this vector.
    std::vector<int> particleBarcodeByIndex;

    std::vector<std::pair<int, int>> vtxToPart;
    std::vector<std::pair<int, int>> partToVtx;

    std::unordered_map<int, int32_t> particlePdgIdByBarcode;
    std::unordered_map<int, int16_t> particleStatusByBarcode;
    // Packed reco::GenStatusFlags from MCTruthHelper, the same helper
    // GenParticleProducer uses, so no barcode-to-reco::GenParticle association is
    // needed. HepMC2 only; the HepMC3 path leaves them 0 until MCTruthHelper grows a
    // HepMC3 specialization.
    std::unordered_map<int, uint16_t> particleStatusFlagsByBarcode;

    [[nodiscard]] bool empty() const { return partBarcodes.empty() && vtxBarcodes.empty(); }
  };

  [[nodiscard]] GenBuild buildFromHepMC2(HepMC::GenEvent const& ev);
  [[nodiscard]] GenBuild buildFromHepMC3(HepMC3::GenEvent const& ev);

}  // namespace truth

#endif
