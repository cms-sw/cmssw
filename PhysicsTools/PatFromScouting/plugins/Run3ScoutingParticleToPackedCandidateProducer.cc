/**
 * Run3ScoutingParticleToPackedCandidateProducer
 *
 * Converts Run3ScoutingParticle to pat::PackedCandidate for MiniAOD compatibility.
 *
 * Requires vertices to be produced first (offlineSlimmedPrimaryVertices).
 */

#include <memory>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

class Run3ScoutingParticleToPackedCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingParticleToPackedCandidateProducer(const edm::ParameterSet&);
  ~Run3ScoutingParticleToPackedCandidateProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<std::vector<Run3ScoutingParticle>> particleToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  const edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> pdtToken_;
  const bool useCHS_;
};

Run3ScoutingParticleToPackedCandidateProducer::Run3ScoutingParticleToPackedCandidateProducer(
    const edm::ParameterSet& iConfig)
    : particleToken_(consumes<std::vector<Run3ScoutingParticle>>(
          iConfig.getParameter<edm::InputTag>("src"))),
      vertexToken_(consumes<reco::VertexCollection>(
          iConfig.getParameter<edm::InputTag>("vertices"))),
      pdtToken_(esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord>()),
      useCHS_(iConfig.getParameter<bool>("CHS")) {
  produces<pat::PackedCandidateCollection>();
}

void Run3ScoutingParticleToPackedCandidateProducer::produce(edm::Event& iEvent,
                                                            const edm::EventSetup& iSetup) {
  auto output = std::make_unique<pat::PackedCandidateCollection>();

  // Get particle data table for mass lookup
  const auto& pdt = iSetup.getData(pdtToken_);

  // Get scouting particles
  const auto& particles = iEvent.get(particleToken_);

  // Get vertices and create RefProd
  auto verticesHandle = iEvent.getHandle(vertexToken_);
  const auto& vertices = *verticesHandle;
  reco::VertexRefProd vertexRefProd(verticesHandle);

  output->reserve(particles.size());

  for (const auto& particle : particles) {
    // Skip pileup particles if CHS is enabled
    if (useCHS_ && particle.vertex() > 0) {
      continue;
    }

    // Get mass from PDT
    const HepPDT::ParticleData* pdtData = pdt.particle(HepPDT::ParticleID(particle.pdgId()));
    if (!pdtData) {
      continue;  // Skip unknown particles
    }
    float mass = pdtData->mass().value();

    // Compute 4-momentum
    float pt = particle.pt();
    float eta = particle.eta();
    float phi = particle.phi();
    float px = pt * std::cos(phi);
    float py = pt * std::sin(phi);
    float pz = pt * std::sinh(eta);
    float energy = std::sqrt(px * px + py * py + pz * pz + mass * mass);
    math::XYZTLorentzVector p4(px, py, pz, energy);

    // Get track kinematics (handle relative vs absolute)
    bool relativeTrackVars = particle.relative_trk_vars();
    float trkPt = relativeTrackVars ? particle.trk_pt() + particle.pt() : particle.trk_pt();
    float trkEta = relativeTrackVars ? particle.trk_eta() + particle.eta() : particle.trk_eta();
    float trkPhi = relativeTrackVars ? particle.trk_phi() + particle.phi() : particle.trk_phi();

    // Determine vertex index (clamp to valid range)
    int vtxIdx = particle.vertex();
    reco::VertexRef::key_type pvKey = 0;
    if (vtxIdx >= 0 && static_cast<size_t>(vtxIdx) < vertices.size()) {
      pvKey = static_cast<reco::VertexRef::key_type>(vtxIdx);
    }

    // Get the associated vertex position
    math::XYZPoint pvPos(0, 0, 0);
    if (pvKey < vertices.size()) {
      pvPos = vertices[pvKey].position();
    }

    // Compute track vertex position from dxy and dz
    // PackedCandidate::packVtx() computes:
    //   dxy = -dxPV * sin(phi+dphi) + dyPV * cos(phi+dphi)
    //   dz = vtx.z - pv.z - (dxPV * cos + dyPV * sin) * pz/pt
    // We invert this to get the vertex position from dxy, dz
    // For simplicity, assume dphi ~ 0 (track phi at vertex ~ calo phi)
    float dxy = particle.dxy();
    float dz = particle.dz();
    float sinPhi = std::sin(phi);
    float cosPhi = std::cos(phi);

    // Assuming the track comes from near the PV in the transverse plane:
    // vtx.x = pv.x - dxy * sin(phi)
    // vtx.y = pv.y + dxy * cos(phi)
    // vtx.z = pv.z + dz (ignoring the small pz/pt correction)
    math::XYZPoint vtxPos(
        pvPos.X() - dxy * sinPhi,
        pvPos.Y() + dxy * cosPhi,
        pvPos.Z() + dz
    );

    // Create PackedCandidate
    // Constructor: (p4, vtx, trkPt, etaAtVtx, phiAtVtx, pdgId, pvRefProd, pvKey)
    pat::PackedCandidate cand(p4, vtxPos, trkPt, trkEta, trkPhi, particle.pdgId(), vertexRefProd, pvKey);

    // Set lost inner hits
    // PackedCandidate::LostInnerHits enum values:
    // validHitInFirstPixelBarrelLayer = -1, noLostInnerHits = 0,
    // oneLostInnerHit = 1, moreLostInnerHits = 2
    pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
    uint8_t scoutingLostHits = particle.lostInnerHits();
    if (scoutingLostHits == 0) {
      // Check if we have valid hit in first pixel barrel (can't determine from scouting)
      lostHits = pat::PackedCandidate::noLostInnerHits;
    } else if (scoutingLostHits == 1) {
      lostHits = pat::PackedCandidate::oneLostInnerHit;
    } else if (scoutingLostHits >= 2) {
      lostHits = pat::PackedCandidate::moreLostInnerHits;
    }
    cand.setLostInnerHits(lostHits);

    // Set track quality (high purity flag)
    // Scouting quality is packed track quality flags
    // Bit 2 (value 4) typically indicates high purity
    int quality = particle.quality();
    bool highPurity = (quality & 4);
    cand.setTrackHighPurity(highPurity);

    // Set PV association quality based on vertex index
    if (vtxIdx == 0) {
      cand.setAssociationQuality(pat::PackedCandidate::UsedInFitTight);
    } else if (vtxIdx > 0) {
      cand.setAssociationQuality(pat::PackedCandidate::CompatibilityDz);
    } else {
      cand.setAssociationQuality(pat::PackedCandidate::NotReconstructedPrimary);
    }

    output->push_back(cand);
  }

  iEvent.put(std::move(output));
}

void Run3ScoutingParticleToPackedCandidateProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker"))
      ->setComment("Input scouting particle collection");
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"))
      ->setComment("Input vertex collection for vertex references");
  desc.add<bool>("CHS", false)->setComment("Apply Charged Hadron Subtraction (skip vtx > 0)");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingParticleToPackedCandidateProducer);
