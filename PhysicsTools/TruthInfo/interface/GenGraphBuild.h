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
#include <unordered_set>
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

  // Contract the parton shower and the intermediate copies of a resonance away. A GEN
  // particle survives if its barcode is in `simContinuedBarcodes` (some SimTrack
  // continues it), or its status is 1, or it is flagged isHardProcess, or it is flagged
  // isLastCopy and is not a parton, diquark, string, cluster or beam pseudoparticle.
  // Ancestry is kept, not cut: every survivor is re-attached through its own production
  // GenVertex to its nearest surviving ancestors, and a GenVertex survives only if it
  // still produces a surviving particle.
  //
  // The isHardProcess and isLastCopy rules read the packed reco::GenStatusFlags, which
  // only buildFromHepMC2 fills; on the HepMC3 path they are 0, so the keep set there
  // degrades to the SIM-continued and status 1 particles.
  // Returns false when the record carried no packed status flags at all, which makes the
  // isHardProcess and isLastCopy rules dead and degrades the keep set to the SIM-continued
  // and status 1 particles. buildFromHepMC3 does not fill them, so that is the HepMC3
  // path; the caller is expected to say so out loud rather than silently ship a keep set
  // that drops every intermediate resonance.
  [[nodiscard]] bool collapseGenShower(GenBuild& gb, std::unordered_set<int> const& simContinuedBarcodes);

  // The barcodes some SimTrack continues, which is the input to the first keep rule
  // above. Templated on the container so this header keeps its HepMC-only dependencies.
  template <typename SimTrackContainerT>
  [[nodiscard]] std::unordered_set<int> simContinuedGenBarcodes(SimTrackContainerT const& tracks) {
    std::unordered_set<int> barcodes;
    barcodes.reserve(tracks.size() * 2);
    for (auto const& track : tracks) {
      if (track.genpartIndex() != -1)
        barcodes.insert(track.genpartIndex());
    }
    return barcodes;
  }

}  // namespace truth

#endif
