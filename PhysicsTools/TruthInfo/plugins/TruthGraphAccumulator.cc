// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// Phase-B (B1): build the mixed (signal + pileup) raw TruthGraph as a
// DigiAccumulatorMixMod, like TrackingTruthAccumulator / CaloTruthAccumulator.
// The framework hands us one sub-event at a time with its NATIVE
// SimTrack/SimVertex/HepMC collections, so trackId/vertIndex/parentIndex are used
// in their original local context (no flattening, no cross-pileup keying); the
// graph does not fragment the way the Phase-A MixCollection prototype did, it is
// identical for standard mixing and premixing, and it is consistent with the
// digis by construction.
//
// GEN handling is configurable per realm, and the same flag means the same thing for
// both:
//   collapsePileupGen (default true) : collapse the GEN decay chain to the stable
//        (status 1) GEN particles on a single gen vertex, keep the SIM continuation
//        (GenToSim links). Compact, and it connects each pileup interaction into one
//        component.
//   collapseSignalGen (default false): keep the full HepMC decay chain, built by the
//        shared truth::GenBuild that TruthGraphProducer uses, so the signal carries
//        intermediate (status 2) particles, GenStatusFlags and the hard-process
//        record. A selection preset seeded on a resonance pdgId needs this: the
//        collapsed form has stable particles only and nothing to seed on.
//
//   collapseGenShower (default true): applies to the full chain above. The parton
//        shower and the intermediate copies of a resonance are contracted away,
//        keeping ancestry, so a resonance that appears several times is one node whose
//        children are its decay products. See truth::collapseGenShower.
//
//   pileupBunchCrossings (default {0} = in-time pileup only): which bunch crossings
//        to include for pileup.
//
// Each node carries an EncodedEventId: (0,0) for the signal, (bunchCrossing,
// pileupIndex) for pileup, so signal and pileup stay distinguishable.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <array>

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"

#include "PhysicsTools/TruthInfo/interface/GenGraphBuild.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {
  // HGCAL Si ADC pulse shape (hgcROCParameters adcPulse), peak at index 2 = the
  // in-time (BX0) readout sample. A hit at bunch crossing bx enters the BX0 sample
  // with the pulse value at offset -bx from its peak, i.e. adcPulse[2 - bx], so
  // out-of-time energy contributes to the digitized amplitude only through the small
  // pulse tails. (Prototype: the SiPM HEback FE has a different shape; one shape is
  // used here for all HGCal calo.)
  constexpr std::array<float, 6> kAdcPulse{{0.00f, 0.017f, 0.817f, 0.163f, 0.003f, 0.000f}};
  float pulseWeight(int bx) {
    const int idx = 2 - bx;
    return (idx >= 0 && idx < static_cast<int>(kAdcPulse.size())) ? kAdcPulse[idx] : 0.f;
  }

  uint64_t packEventId(EncodedEventId const& id) {
    // EncodedEventId is a single uint32 rawId; use the typed accessor rather than a
    // byte copy so the key stays portable and cannot pick up a future member/padding.
    static_assert(sizeof(EncodedEventId) == sizeof(uint32_t));
    return static_cast<uint64_t>(id.rawId());
  }

  // Stable (status 1) GEN particles as (barcode, pdgId). Used to collapse the GEN
  // part to "stable particles on a single gen vertex".
  std::vector<std::pair<int, int>> stableFromHepMC2(HepMC::GenEvent const& ev) {
    std::vector<std::pair<int, int>> out;
    for (auto p = ev.particles_begin(); p != ev.particles_end(); ++p) {
      if (*p != nullptr && (*p)->status() == 1)
        out.emplace_back((*p)->barcode(), (*p)->pdg_id());
    }
    return out;
  }

  std::vector<std::pair<int, int>> stableFromHepMC3(HepMC3::GenEvent const& ev) {
    std::vector<std::pair<int, int>> out;
    for (auto const& p : ev.particles()) {
      if (p && p->status() == 1)
        out.emplace_back(p->id(), p->pid());
    }
    return out;
  }

  // Read the stable GEN particles from a signal Event or a PileUpEventPrincipal
  // (both expose getByLabel), preferring HepMC3.
  template <class EvT>
  std::vector<std::pair<int, int>> readStableGen(EvT const& ev,
                                                 edm::InputTag const& hepmc3Tag,
                                                 edm::InputTag const& hepmc2Tag) {
    edm::Handle<edm::HepMC3Product> h3;
    if (ev.getByLabel(hepmc3Tag, h3) && h3.isValid() && h3->GetEvent() != nullptr) {
      HepMC3::GenEvent ev3;
      ev3.read_data(*h3->GetEvent());
      return stableFromHepMC3(ev3);
    }
    edm::Handle<edm::HepMCProduct> h2;
    if (ev.getByLabel(hepmc2Tag, h2) && h2.isValid() && h2->GetEvent() != nullptr)
      return stableFromHepMC2(*h2->GetEvent());
    return {};
  }

  // The full HepMC record for one sub-event, in the same flattened form
  // TruthGraphProducer builds from an unmixed event, optionally with the parton
  // shower and the intermediate resonance copies contracted away.
  template <class EvT>
  truth::GenBuild readFullGen(EvT const& ev,
                              edm::InputTag const& hepmc3Tag,
                              edm::InputTag const& hepmc2Tag,
                              bool collapseShower,
                              edm::SimTrackContainer const& tracks,
                              bool& degradedCollapseWarned) {
    truth::GenBuild gb;
    edm::Handle<edm::HepMC3Product> h3;
    if (ev.getByLabel(hepmc3Tag, h3) && h3.isValid() && h3->GetEvent() != nullptr) {
      HepMC3::GenEvent ev3;
      ev3.read_data(*h3->GetEvent());
      gb = truth::buildFromHepMC3(ev3);
    } else {
      edm::Handle<edm::HepMCProduct> h2;
      if (ev.getByLabel(hepmc2Tag, h2) && h2.isValid() && h2->GetEvent() != nullptr)
        gb = truth::buildFromHepMC2(*h2->GetEvent());
    }
    if (collapseShower && !gb.empty()) {
      // The degraded path is a property of the sample, not of one sub-event, so one
      // warning per stream says everything 200 per event would.
      if (!truth::collapseGenShower(gb, truth::simContinuedGenBarcodes(tracks)) && !degradedCollapseWarned) {
        degradedCollapseWarned = true;
        edm::LogWarning("TruthGraphAccumulator")
            << "collapseGenShower ran on a GEN record with no packed status flags, which "
               "buildFromHepMC3 does not fill. The isHardProcess and isLastCopy keep rules "
               "are then dead and every intermediate resonance is dropped, so a selection "
               "preset seeded on a resonance pdgId will match nothing. Set "
               "collapseGenShower=False on a HepMC3 sample.";
      }
    }
    return gb;
  }
}  // namespace

class TruthGraphAccumulator : public DigiAccumulatorMixMod {
public:
  TruthGraphAccumulator(edm::ParameterSet const&, edm::ProducesCollector, edm::ConsumesCollector&);

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;

private:
  // Append one sub-event. SimTrack/SimVertex ids are local to this sub-event.
  // The GEN half is `fullGen` when that is non-null and non-empty, otherwise the
  // collapsed `stableGen` when that is non-empty, otherwise absent. Either GEN form
  // is linked to the primary SimTracks by GenToSim edges. `genEvent` identifies the
  // sub-event the GEN nodes belong to.
  void addSubEvent(std::vector<std::pair<int, int>> const& stableGen,
                   truth::GenBuild const* fullGen,
                   edm::SimTrackContainer const& tracks,
                   edm::SimVertexContainer const& vertices,
                   EncodedEventId const& eid,
                   int32_t genEvent);

  // Append this sub-event's sim-hits to the merged collections, re-tagged with `eid`
  // so they carry per-interaction provenance (native hits are all tagged (0,0)).
  template <class EvT>
  void addSubEventHits(EvT const& ev, EncodedEventId const& eid);

  // Merge one sim-hit collection family (PCaloHit or PSimHit) from the sub-event,
  // re-tagging each hit's eventId. Kept per subdetector family so a downstream
  // consumer can apply the right sim-to-reco DetId relabelling per collection.
  template <class HitT, class EvT>
  void mergeHits(EvT const& ev,
                 std::vector<edm::InputTag> const& tags,
                 EncodedEventId const& eid,
                 std::vector<HitT>& out);

  // Prototype energy-budget closure: sum this sub-event's HGCal calo hit energy per
  // raw sim DetId, both raw and weighted by the ADC pulse shape at this bunch
  // crossing (pulseWeight(bx)). Called for EVERY sub-event (signal + all bunch
  // crossings, before the in-time keepBx filter), so the totals carry the
  // out-of-time pileup the per-particle graph deliberately leaves out; the
  // pulse-weighted total is the digitized-amplitude proxy that matches reco.
  template <class EvT>
  void accumulateCellEnergy(EvT const& ev, int bx);

  const edm::InputTag simTrackTag_;
  const edm::InputTag simVertexTag_;
  const edm::InputTag hepmc3Tag_;
  const edm::InputTag hepmc2Tag_;
  const std::vector<edm::InputTag> caloHitTags_;
  const std::vector<edm::InputTag> ecalHitTags_;
  const std::vector<edm::InputTag> hcalHitTags_;
  const std::vector<edm::InputTag> trackerHitTags_;
  const std::vector<edm::InputTag> muonHitTags_;
  const std::vector<edm::InputTag> mtdHitTags_;
  const std::vector<int> pileupBunchCrossings_;
  const bool collapsePileupGen_;
  const bool collapseSignalGen_;
  const bool collapseGenShower_;
  const bool computeCellEnergyBudget_;

  int pileupCount_ = 0;
  // Warn once PER COLLECTION, not once overall: a single shared flag reports only the
  // first collection that goes missing and hides every later one, so a real premix
  // problem in the calorimeter can be masked by an unrelated tracker collection.
  std::set<std::string> missingHitsWarned_;
  bool degradedCollapseWarned_ = false;

  // Merged calorimeter sim-hits across signal + kept pileup, each re-tagged with its
  // sub-event EncodedEventId so the (eventId,trackId) hit-index key resolves pileup
  // nodes at RECO (the native pileup hits are consumed transiently here). Kept one
  // vector per subdetector family so the relabelling at RECO stays per collection.
  std::vector<PCaloHit> mergedCaloHits_;
  std::vector<PCaloHit> mergedEcalHits_;
  std::vector<PCaloHit> mergedHcalHits_;
  // Tracking sim-hits (tracker, muon chambers, MTD) as PSimHit, same per-interaction
  // re-tagging. Tracker pileup is by far the largest family; see the customise note.
  std::vector<PSimHit> mergedTrackerHits_;
  std::vector<PSimHit> mergedMuonHits_;
  std::vector<PSimHit> mergedMtdHits_;

  // Prototype: per-cell HGCal calorimeter energy summed over ALL bunch crossings
  // (in-time + out-of-time), keyed by raw sim DetId. cellTotalEnergy_ is the raw
  // sum; cellWeightedEnergy_ weights each hit by the ADC pulse shape at its bunch
  // crossing (pulseWeight), so out-of-time enters only through the pulse tails and
  // the total is a proxy for the digitized BX0 amplitude. The energy-budget closure:
  // "untracked" = this total minus the energy on retained in-time truth branches.
  std::unordered_map<uint32_t, float> cellTotalEnergy_;
  std::unordered_map<uint32_t, float> cellWeightedEnergy_;
  // In-time (bx 0) pulse-weighted energy per cell: the reference the in-time truth
  // graph can track. A cell with all-bx energy but no in-time energy is pure
  // out-of-time; a cell with no all-bx energy at all is pure noise.
  std::unordered_map<uint32_t, float> cellInTimeEnergy_;

  // Rejected GenToSim links, summed over the event's sub-events: a SimTrack whose
  // genpartIndex resolves to a GEN particle of a different pdgId is not that
  // particle's continuation, so the link is dropped rather than written wrong.
  unsigned int rejectedGenToSimLinks_ = 0;

  std::vector<TruthGraph::NodeRef> nodes_;
  std::vector<int32_t> pdgId_;
  std::vector<int16_t> status_;
  std::vector<uint16_t> statusFlags_;
  std::vector<int32_t> genEventOfNode_;
  std::vector<uint64_t> eventId_;
  std::vector<int32_t> simTrackToVtx_;
  std::vector<int32_t> simTrackToGen_;
  std::vector<std::pair<uint32_t, uint32_t>> edges_;
  std::vector<uint8_t> edgeKinds_;
  std::vector<uint16_t> simVertexProcessType_;  // node-parallel; G4 process subtype (SimVertex only)
  std::vector<uint8_t> simTrackBackscattered_;  // node-parallel; albedo flag (SimTrack only)

  [[nodiscard]] bool keepBx(int bx) const {
    return std::find(pileupBunchCrossings_.begin(), pileupBunchCrossings_.end(), bx) != pileupBunchCrossings_.end();
  }
};

TruthGraphAccumulator::TruthGraphAccumulator(edm::ParameterSet const& cfg,
                                             edm::ProducesCollector producesCollector,
                                             edm::ConsumesCollector& iC)
    : simTrackTag_(cfg.getParameter<edm::InputTag>("simTracks")),
      simVertexTag_(cfg.getParameter<edm::InputTag>("simVertices")),
      hepmc3Tag_(cfg.getParameter<edm::InputTag>("genEventHepMC3")),
      hepmc2Tag_(cfg.getParameter<edm::InputTag>("genEventHepMC")),
      caloHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("caloHits")),
      ecalHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("ecalHits")),
      hcalHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("hcalHits")),
      trackerHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("trackerHits")),
      muonHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("muonHits")),
      mtdHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("mtdHits")),
      pileupBunchCrossings_(cfg.getParameter<std::vector<int>>("pileupBunchCrossings")),
      collapsePileupGen_(cfg.getParameter<bool>("collapsePileupGen")),
      collapseSignalGen_(cfg.getParameter<bool>("collapseSignalGen")),
      collapseGenShower_(cfg.getParameter<bool>("collapseGenShower")),
      computeCellEnergyBudget_(
          cfg.existsAs<bool>("computeCellEnergyBudget") ? cfg.getParameter<bool>("computeCellEnergyBudget") : false) {
  producesCollector.produces<TruthGraph>();
  producesCollector.produces<std::vector<PCaloHit>>("mergedHGCHits");
  producesCollector.produces<std::vector<PCaloHit>>("mergedEcalHits");
  producesCollector.produces<std::vector<PCaloHit>>("mergedHcalHits");
  producesCollector.produces<std::vector<PSimHit>>("mergedTrackerHits");
  producesCollector.produces<std::vector<PSimHit>>("mergedMuonHits");
  producesCollector.produces<std::vector<PSimHit>>("mergedMtdHits");
  if (computeCellEnergyBudget_) {
    producesCollector.produces<std::vector<unsigned int>>("cellTotalDetId");
    producesCollector.produces<std::vector<float>>("cellTotalEnergy");
    producesCollector.produces<std::vector<float>>("cellInTimeEnergy");
  }
  iC.consumes<edm::SimTrackContainer>(simTrackTag_);
  iC.consumes<edm::SimVertexContainer>(simVertexTag_);
  iC.mayConsume<edm::HepMC3Product>(hepmc3Tag_);
  iC.mayConsume<edm::HepMCProduct>(hepmc2Tag_);
  for (auto const* tags : {&caloHitTags_, &ecalHitTags_, &hcalHitTags_})
    for (auto const& tag : *tags)
      iC.mayConsume<std::vector<PCaloHit>>(tag);
  for (auto const* tags : {&trackerHitTags_, &muonHitTags_, &mtdHitTags_})
    for (auto const& tag : *tags)
      iC.mayConsume<std::vector<PSimHit>>(tag);
}

void TruthGraphAccumulator::initializeEvent(edm::Event const&, edm::EventSetup const&) {
  pileupCount_ = 0;
  rejectedGenToSimLinks_ = 0;
  mergedCaloHits_.clear();
  mergedEcalHits_.clear();
  mergedHcalHits_.clear();
  mergedTrackerHits_.clear();
  mergedMuonHits_.clear();
  mergedMtdHits_.clear();
  nodes_.clear();
  pdgId_.clear();
  status_.clear();
  statusFlags_.clear();
  genEventOfNode_.clear();
  eventId_.clear();
  simTrackToVtx_.clear();
  simTrackToGen_.clear();
  edges_.clear();
  edgeKinds_.clear();
  simVertexProcessType_.clear();
  simTrackBackscattered_.clear();
  cellTotalEnergy_.clear();
  cellWeightedEnergy_.clear();
  cellInTimeEnergy_.clear();
}

void TruthGraphAccumulator::addSubEvent(std::vector<std::pair<int, int>> const& stableGen,
                                        truth::GenBuild const* fullGen,
                                        edm::SimTrackContainer const& tracks,
                                        edm::SimVertexContainer const& vertices,
                                        EncodedEventId const& eid,
                                        int32_t genEvent) {
  const uint64_t packed = packEventId(eid);
  auto pushNode = [&](TruthGraph::NodeKind kind, int64_t key, int32_t pdg, int16_t st) {
    const uint32_t node = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(TruthGraph::NodeRef{kind, key});
    pdgId_.push_back(pdg);
    status_.push_back(st);
    statusFlags_.push_back(0);
    genEventOfNode_.push_back(-1);
    eventId_.push_back(packed);
    simTrackToVtx_.push_back(-1);
    simTrackToGen_.push_back(-1);
    simVertexProcessType_.push_back(0);
    simTrackBackscattered_.push_back(0);
    return node;
  };
  auto pushEdge = [&](uint32_t src, uint32_t dst, TruthGraph::EdgeKind k) {
    edges_.emplace_back(src, dst);
    edgeKinds_.push_back(static_cast<uint8_t>(k));
  };

  // GEN realm, one of two forms. Either way genBarcodeToNode maps a HepMC barcode to
  // its GenParticle node, which is what GenToSim linking below needs.
  std::unordered_map<int, uint32_t> genBarcodeToNode;
  const bool useFullGen = (fullGen != nullptr && !fullGen->empty());

  if (useFullGen) {
    // Full HepMC decay chain: every particle at its own status, both Gen edge
    // directions, and the GenEvent node attached to the vertices with no incoming
    // particle so the component has a single source.
    const uint32_t genEventNode = pushNode(TruthGraph::NodeKind::GenEvent, static_cast<int64_t>(genEvent), 0, 0);
    genEventOfNode_[genEventNode] = genEvent;

    std::unordered_map<int, uint32_t> genVtxBarcodeToNode;
    genVtxBarcodeToNode.reserve(fullGen->vtxBarcodes.size() * 2);
    for (int vbc : fullGen->vtxBarcodes) {
      const uint32_t vn = pushNode(TruthGraph::NodeKind::GenVertex, static_cast<int64_t>(vbc), 0, 0);
      genEventOfNode_[vn] = genEvent;
      genVtxBarcodeToNode.emplace(vbc, vn);
    }

    genBarcodeToNode.reserve(fullGen->partBarcodes.size() * 2);
    for (int pbc : fullGen->partBarcodes) {
      const auto itPdg = fullGen->particlePdgIdByBarcode.find(pbc);
      const auto itStatus = fullGen->particleStatusByBarcode.find(pbc);
      const int32_t pdg = (itPdg != fullGen->particlePdgIdByBarcode.end()) ? itPdg->second : 0;
      const int16_t st = (itStatus != fullGen->particleStatusByBarcode.end()) ? itStatus->second : 0;
      const uint32_t pn = pushNode(TruthGraph::NodeKind::GenParticle, static_cast<int64_t>(pbc), pdg, st);
      const auto itFlags = fullGen->particleStatusFlagsByBarcode.find(pbc);
      if (itFlags != fullGen->particleStatusFlagsByBarcode.end())
        statusFlags_[pn] = itFlags->second;
      genEventOfNode_[pn] = genEvent;
      genBarcodeToNode.emplace(pbc, pn);
    }

    std::unordered_map<int, unsigned int> vtxIncoming;
    for (auto const& [pbc, vbc] : fullGen->partToVtx)
      ++vtxIncoming[vbc];

    for (auto const& [vbc, pbc] : fullGen->vtxToPart) {
      auto itV = genVtxBarcodeToNode.find(vbc);
      auto itP = genBarcodeToNode.find(pbc);
      if (itV != genVtxBarcodeToNode.end() && itP != genBarcodeToNode.end())
        pushEdge(itV->second, itP->second, TruthGraph::EdgeKind::Gen);
    }
    for (auto const& [pbc, vbc] : fullGen->partToVtx) {
      auto itP = genBarcodeToNode.find(pbc);
      auto itV = genVtxBarcodeToNode.find(vbc);
      if (itP != genBarcodeToNode.end() && itV != genVtxBarcodeToNode.end())
        pushEdge(itP->second, itV->second, TruthGraph::EdgeKind::Gen);
    }

    // Attach the GenEvent node PER CONNECTED COMPONENT, which is what TruthGraphProducer
    // does on an unmixed event and what GenGraphBuild.h requires of both: a component
    // whose vertices all have an incoming particle has no source of its own, and a
    // single sub-event-wide root count would let one component with a source suppress
    // the fallback for another that has none, leaving the second unreachable.
    // A collider record is entirely the fallback case: the beam particles give the first
    // vertex an incoming particle, so no vertex is a source.
    std::unordered_map<int, int> componentOfVtx;
    {
      // Two vertices are in the same component when a particle touches both.
      std::unordered_map<int, std::vector<int>> partAdjacency;
      std::unordered_map<int, std::vector<int>> vtxNeighbours;
      for (auto const& [vbc, pbc] : fullGen->vtxToPart)
        partAdjacency[pbc].push_back(vbc);
      for (auto const& [pbc, vbc] : fullGen->partToVtx)
        partAdjacency[pbc].push_back(vbc);
      for (auto const& [pbc, vtxs] : partAdjacency) {
        for (std::size_t i = 1; i < vtxs.size(); ++i) {
          vtxNeighbours[vtxs[0]].push_back(vtxs[i]);
          vtxNeighbours[vtxs[i]].push_back(vtxs[0]);
        }
      }

      int nextComponent = 0;
      std::vector<int> stack;
      for (int vbc : fullGen->vtxBarcodes) {
        if (componentOfVtx.count(vbc) != 0)
          continue;
        const int component = nextComponent++;
        stack.push_back(vbc);
        componentOfVtx.emplace(vbc, component);
        while (!stack.empty()) {
          const int current = stack.back();
          stack.pop_back();
          const auto it = vtxNeighbours.find(current);
          if (it == vtxNeighbours.end())
            continue;
          for (const int next : it->second) {
            if (componentOfVtx.emplace(next, component).second)
              stack.push_back(next);
          }
        }
      }
    }

    // Residual gap, shared with TruthGraphProducer so the two stay consistent: source
    // counting is per undirected component, but reachability from the GenEvent node is
    // DIRECTED. A component containing both a true source and a beam-fed branch would
    // attach only the source and leave the branch unreachable. No current record mixes
    // the two in one component: a collider record is wholly sourceless and a gun record
    // wholly source-rooted.
    std::unordered_map<int, unsigned int> rootsInComponent;
    for (int vbc : fullGen->vtxBarcodes) {
      if (vtxIncoming[vbc] == 0)
        ++rootsInComponent[componentOfVtx.at(vbc)];
    }
    for (int vbc : fullGen->vtxBarcodes) {
      const bool isSource = vtxIncoming[vbc] == 0;
      const bool componentHasNoSource = rootsInComponent[componentOfVtx.at(vbc)] == 0;
      if (isSource || componentHasNoSource)
        pushEdge(genEventNode, genVtxBarcodeToNode.at(vbc), TruthGraph::EdgeKind::Gen);
    }
  } else if (!stableGen.empty()) {
    // Collapsed GEN: one gen vertex + the stable gen particles.
    const uint32_t genVtxNode = pushNode(TruthGraph::NodeKind::GenVertex, 0, 0, 0);
    genEventOfNode_[genVtxNode] = genEvent;
    genBarcodeToNode.reserve(stableGen.size() * 2);
    for (auto const& [barcode, pdg] : stableGen) {
      const uint32_t pn = pushNode(TruthGraph::NodeKind::GenParticle, barcode, pdg, 1);
      genEventOfNode_[pn] = genEvent;
      pushEdge(genVtxNode, pn, TruthGraph::EdgeKind::Gen);
      genBarcodeToNode.emplace(barcode, pn);
    }
  }

  // SIM realm (native local ids).
  std::unordered_map<uint32_t, uint32_t> vertexIdToNode;
  vertexIdToNode.reserve(vertices.size() * 2);
  const uint32_t baseVtx = static_cast<uint32_t>(nodes_.size());
  for (auto const& v : vertices) {
    const uint32_t node = pushNode(TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(v.vertexId()), 0, 0);
    simVertexProcessType_[node] = static_cast<uint16_t>(v.processType());
    vertexIdToNode.emplace(static_cast<uint32_t>(v.vertexId()), node);
  }
  const uint32_t baseTrk = static_cast<uint32_t>(nodes_.size());
  std::unordered_map<uint32_t, uint32_t> trackIdToNode;
  trackIdToNode.reserve(tracks.size() * 2);
  for (auto const& t : tracks) {
    const uint32_t node = pushNode(TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(t.trackId()), t.type(), 0);
    simTrackBackscattered_[node] = t.isFromBackScattering() ? 1 : 0;
    trackIdToNode.emplace(t.trackId(), node);
  }

  // Production edge: track.vertIndex() is the local vector index into `vertices`.
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const int vi = tracks[i].vertIndex();
    if (vi < 0 || static_cast<std::size_t>(vi) >= vertices.size())
      continue;
    const uint32_t trkNode = baseTrk + static_cast<uint32_t>(i);
    const uint32_t prodVtxNode = baseVtx + static_cast<uint32_t>(vi);
    pushEdge(prodVtxNode, trkNode, TruthGraph::EdgeKind::Sim);
    simTrackToVtx_[trkNode] = static_cast<int32_t>(prodVtxNode);
  }

  // Decay edge: vertex.parentIndex() is the trackId of the parent track.
  for (auto const& v : vertices) {
    if (v.parentIndex() < 0)
      continue;
    auto pIt = trackIdToNode.find(static_cast<uint32_t>(v.parentIndex()));
    auto vIt = vertexIdToNode.find(static_cast<uint32_t>(v.vertexId()));
    if (pIt != trackIdToNode.end() && vIt != vertexIdToNode.end())
      pushEdge(pIt->second, vIt->second, TruthGraph::EdgeKind::Sim);
  }

  // GenToSim: a primary SimTrack's genpartIndex is its GEN particle's barcode. The
  // two must agree on pdgId, otherwise the barcode does not identify this track's
  // generator particle and no link is written.
  if (!genBarcodeToNode.empty()) {
    for (auto const& t : tracks) {
      auto gIt = genBarcodeToNode.find(t.genpartIndex());
      if (gIt == genBarcodeToNode.end())
        continue;
      auto sIt = trackIdToNode.find(t.trackId());
      if (sIt == trackIdToNode.end())
        continue;
      if (pdgId_[gIt->second] != t.type()) {
        ++rejectedGenToSimLinks_;
        continue;
      }
      pushEdge(gIt->second, sIt->second, TruthGraph::EdgeKind::GenToSim);
      simTrackToGen_[sIt->second] = static_cast<int32_t>(gIt->second);
    }
  }
}

template <class HitT, class EvT>
void TruthGraphAccumulator::mergeHits(EvT const& ev,
                                      std::vector<edm::InputTag> const& tags,
                                      EncodedEventId const& eid,
                                      std::vector<HitT>& out) {
  for (auto const& tag : tags) {
    edm::Handle<std::vector<HitT>> hits;
    ev.getByLabel(tag, hits);
    if (!hits.isValid()) {
      // State the fact, then the two things that cause it. Naming premixing as THE cause
      // is wrong and actively misleading: a collection configured here but absent from
      // the running geometry, for instance the strip tracker under Run4, produces the
      // same invalid handle and nothing is wrong.
      if (missingHitsWarned_.insert(tag.encode()).second) {
        edm::LogWarning("TruthGraphAccumulator")
            << "sim-hit collection " << tag.encode() << " not found in a sub-event, so it contributes no truth hits."
            << " Either this collection does not exist in the running geometry, or the pileup is premixed and its"
            << " sim-hits were digitized away; pileup-aware truth needs classic (non-premixed) pileup.";
      }
      continue;
    }
    out.reserve(out.size() + hits->size());
    for (HitT hit : *hits) {  // copy: re-tag the eventId to this sub-event
      hit.setEventId(eid);
      out.push_back(hit);
    }
  }
}

template <class EvT>
void TruthGraphAccumulator::addSubEventHits(EvT const& ev, EncodedEventId const& eid) {
  mergeHits(ev, caloHitTags_, eid, mergedCaloHits_);
  mergeHits(ev, ecalHitTags_, eid, mergedEcalHits_);
  mergeHits(ev, hcalHitTags_, eid, mergedHcalHits_);
  mergeHits(ev, trackerHitTags_, eid, mergedTrackerHits_);
  mergeHits(ev, muonHitTags_, eid, mergedMuonHits_);
  mergeHits(ev, mtdHitTags_, eid, mergedMtdHits_);
}

template <class EvT>
void TruthGraphAccumulator::accumulateCellEnergy(EvT const& ev, int bx) {
  // HGCal calorimeter only (the TICL-relevant calo) for this prototype, summed by
  // raw sim DetId. hit.id() is the sim DetId, hit.energy() the deposited energy; the
  // pulse-shape weight for this bunch crossing down-weights out-of-time deposits the
  // way the shaper does.
  const float w = pulseWeight(bx);
  for (auto const& tag : caloHitTags_) {
    edm::Handle<std::vector<PCaloHit>> hits;
    ev.getByLabel(tag, hits);
    if (!hits.isValid())
      continue;
    for (auto const& hit : *hits) {
      cellTotalEnergy_[hit.id()] += hit.energy();
      cellWeightedEnergy_[hit.id()] += hit.energy() * w;
      if (bx == 0)
        cellInTimeEnergy_[hit.id()] += hit.energy() * w;
    }
  }
}

void TruthGraphAccumulator::accumulate(edm::Event const& event, edm::EventSetup const&) {
  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  event.getByLabel(simTrackTag_, tracks);
  event.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;
  std::vector<std::pair<int, int>> stableGen;
  truth::GenBuild fullGen;
  if (collapseSignalGen_)
    stableGen = readStableGen(event, hepmc3Tag_, hepmc2Tag_);
  else
    fullGen = readFullGen(event, hepmc3Tag_, hepmc2Tag_, collapseGenShower_, *tracks, degradedCollapseWarned_);
  const EncodedEventId sigEid(0, 0);
  addSubEvent(stableGen, &fullGen, *tracks, *vertices, sigEid, 0);
  addSubEventHits(event, sigEid);
  if (computeCellEnergyBudget_)
    accumulateCellEnergy(event, 0);  // signal is in-time (bx 0)
}

void TruthGraphAccumulator::accumulate(PileUpEventPrincipal const& pep, edm::EventSetup const&, edm::StreamID const&) {
  const int bx = pep.bunchCrossing();
  // Sum the per-cell energy over EVERY bunch crossing, before the in-time filter, so
  // the budget total carries the out-of-time pileup the per-particle graph drops.
  if (computeCellEnergyBudget_)
    accumulateCellEnergy(pep, bx);
  if (!keepBx(bx))
    return;

  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  pep.getByLabel(simTrackTag_, tracks);
  pep.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;

  std::vector<std::pair<int, int>> stableGen;
  truth::GenBuild fullGen;
  if (collapsePileupGen_)
    stableGen = readStableGen(pep, hepmc3Tag_, hepmc2Tag_);
  else
    fullGen = readFullGen(pep, hepmc3Tag_, hepmc2Tag_, collapseGenShower_, *tracks, degradedCollapseWarned_);

  // Global counter across bunch crossings: EncodedEventId stores abs(bx), so a
  // per-bx counter would give (-1,1) and (+1,1) identical packed ids. A single
  // counter keeps every pileup interaction's tag unique regardless of bx sign.
  const int puIndex = ++pileupCount_;
  // EncodedEventId packs the event number into 16 bits; an unrealistic pileup
  // multiplicity would overflow into the bunch-crossing bits and alias ids.
  if (puIndex > 0xFFFF)
    throw cms::Exception("TruthGraphAccumulator")
        << "pileup sub-event count " << puIndex << " exceeds the 16-bit EncodedEventId event field";
  const EncodedEventId puEid(bx, puIndex);
  addSubEvent(stableGen, &fullGen, *tracks, *vertices, puEid, puIndex);
  addSubEventHits(pep, puEid);
}

void TruthGraphAccumulator::finalizeEvent(edm::Event& event, edm::EventSetup const&) {
  auto out = std::make_unique<TruthGraph>();
  const uint32_t nNodes = static_cast<uint32_t>(nodes_.size());

  out->nodes() = std::move(nodes_);
  out->pdgId() = std::move(pdgId_);
  out->status() = std::move(status_);
  out->eventId() = std::move(eventId_);
  out->simTrackToVtx() = std::move(simTrackToVtx_);
  out->simTrackToGen() = std::move(simTrackToGen_);
  out->simVertexProcessType() = std::move(simVertexProcessType_);
  out->simTrackBackscattered() = std::move(simTrackBackscattered_);
  out->statusFlags() = std::move(statusFlags_);
  out->genEventOfNode() = std::move(genEventOfNode_);
  out->simVtxToGen().assign(nNodes, -1);

  if (rejectedGenToSimLinks_ != 0) {
    edm::LogWarning("TruthGraphAccumulator")
        << rejectedGenToSimLinks_ << " GenToSim links dropped in this event because the SimTrack pdgId disagreed with"
        << " the GEN particle its genpartIndex points at.";
  }

  // CSR out-edges via the counting-sort cursor scatter: each edge lands in its
  // source's range, by construction (no sort, no permutation vector).
  out->offsets().assign(nNodes + 1, 0);
  for (auto const& e : edges_)
    ++out->offsets()[e.first + 1];
  for (uint32_t i = 1; i <= nNodes; ++i)
    out->offsets()[i] += out->offsets()[i - 1];

  out->edges().resize(edges_.size());
  out->edgeKind().resize(edges_.size());
  std::vector<uint32_t> cursor = out->offsets();
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const uint32_t pos = cursor[edges_[e].first]++;
    out->edges()[pos] = edges_[e].second;
    out->edgeKind()[pos] = edgeKinds_[e];
  }

  if (!out->isConsistent())
    throw cms::Exception("TruthGraphAccumulator") << "Produced TruthGraph is not consistent";

  event.put(std::move(out));

  if (computeCellEnergyBudget_) {
    // Compute before mergedCaloHits_ is moved below. Persist the pulse-weighted total
    // (the digitized-amplitude proxy that matches reco); log the raw and the weighted
    // out-of-time fraction so the pulse-shape suppression of out-of-time is visible.
    auto detIds = std::make_unique<std::vector<unsigned int>>();
    auto energies = std::make_unique<std::vector<float>>();
    auto inTime = std::make_unique<std::vector<float>>();  // parallel to detIds
    detIds->reserve(cellWeightedEnergy_.size());
    energies->reserve(cellWeightedEnergy_.size());
    inTime->reserve(cellWeightedEnergy_.size());
    double allBxWeighted = 0.;
    for (auto const& [detId, energy] : cellWeightedEnergy_) {
      detIds->push_back(detId);
      energies->push_back(energy);
      auto const it = cellInTimeEnergy_.find(detId);
      inTime->push_back(it == cellInTimeEnergy_.end() ? 0.f : it->second);
      allBxWeighted += energy;
    }
    double allBxRaw = 0.;
    for (auto const& [detId, energy] : cellTotalEnergy_)
      allBxRaw += energy;
    // In-time (bx 0) energy is what went into the kept merged collection; its
    // digitized weight is pulseWeight(0).
    double inTimeRaw = 0.;
    for (auto const& hit : mergedCaloHits_)
      inTimeRaw += hit.energy();
    const double inTimeWeighted = inTimeRaw * pulseWeight(0);
    const double untrackedRaw = allBxRaw - inTimeRaw;
    const double untrackedWeighted = allBxWeighted - inTimeWeighted;
    edm::LogPrint("TruthGraphAccumulator")
        << "cell energy budget (HGCal): cells=" << cellWeightedEnergy_.size() << " | raw: allBx=" << allBxRaw
        << " inTime=" << inTimeRaw << " untrackedFraction=" << (allBxRaw > 0. ? untrackedRaw / allBxRaw : 0.)
        << " | pulse-weighted: allBx=" << allBxWeighted << " inTime=" << inTimeWeighted
        << " untrackedFraction=" << (allBxWeighted > 0. ? untrackedWeighted / allBxWeighted : 0.);
    event.put(std::move(detIds), "cellTotalDetId");
    event.put(std::move(energies), "cellTotalEnergy");
    event.put(std::move(inTime), "cellInTimeEnergy");
  }

  event.put(std::make_unique<std::vector<PCaloHit>>(std::move(mergedCaloHits_)), "mergedHGCHits");
  event.put(std::make_unique<std::vector<PCaloHit>>(std::move(mergedEcalHits_)), "mergedEcalHits");
  event.put(std::make_unique<std::vector<PCaloHit>>(std::move(mergedHcalHits_)), "mergedHcalHits");
  event.put(std::make_unique<std::vector<PSimHit>>(std::move(mergedTrackerHits_)), "mergedTrackerHits");
  event.put(std::make_unique<std::vector<PSimHit>>(std::move(mergedMuonHits_)), "mergedMuonHits");
  event.put(std::make_unique<std::vector<PSimHit>>(std::move(mergedMtdHits_)), "mergedMtdHits");
}

DEFINE_DIGI_ACCUMULATOR(TruthGraphAccumulator);
