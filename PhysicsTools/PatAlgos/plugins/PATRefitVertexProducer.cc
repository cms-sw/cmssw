/**
  \class PATRefitVertexProducer

  This producer is intended to take packedCandidates with tracks associated to
  the PV and refit the PV (applying or not) BeamSpot constraint
  
  \autor Michal Bluj, NCBJ Warsaw (and then others)

 **/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include <memory>

class PATRefitVertexProducer : public edm::stream::EDProducer<> {
public:
  explicit PATRefitVertexProducer(const edm::ParameterSet&);
  ~PATRefitVertexProducer() override {}

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  //--- utility methods

  //--- configuration parameters
  edm::EDGetTokenT<std::vector<pat::PackedCandidate> > srcCands_, srcLostTracks_, srcEleKfTracks_;
  edm::EDGetTokenT<reco::VertexCollection> srcVertices_;
  edm::EDGetTokenT<reco::BeamSpot> srcBeamSpot_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transTrackBuilderToken_;
  bool useBeamSpot_;
  bool useLostTracks_;
  bool useEleKfTracks_;
};

PATRefitVertexProducer::PATRefitVertexProducer(const edm::ParameterSet& cfg)
    : srcCands_(consumes<std::vector<pat::PackedCandidate> >(cfg.getParameter<edm::InputTag>("srcCands"))),
      srcLostTracks_(consumes<std::vector<pat::PackedCandidate> >(cfg.getParameter<edm::InputTag>("srcLostTracks"))),
      srcEleKfTracks_(consumes<std::vector<pat::PackedCandidate> >(cfg.getParameter<edm::InputTag>("srcEleKfTracks"))),
      srcVertices_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("srcVertices"))),
      srcBeamSpot_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("srcBeamSpot"))),
      transTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      useBeamSpot_(cfg.getParameter<bool>("useBeamSpot")),
      useLostTracks_(cfg.getParameter<bool>("useLostTracks")),
      useEleKfTracks_(cfg.getParameter<bool>("useEleKfTracks")) {
  produces<reco::VertexCollection>();
}

void PATRefitVertexProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Obtain collections
  edm::Handle<std::vector<pat::PackedCandidate> > cands;
  evt.getByToken(srcCands_, cands);

  edm::Handle<std::vector<pat::PackedCandidate> > lostTrackCands;
  if (useLostTracks_)
    evt.getByToken(srcLostTracks_, lostTrackCands);

  edm::Handle<std::vector<pat::PackedCandidate> > eleKfTrackCands;
  if (useEleKfTracks_)
    evt.getByToken(srcEleKfTracks_, eleKfTrackCands);

  edm::Handle<reco::VertexCollection> vertices;
  evt.getByToken(srcVertices_, vertices);
  const reco::Vertex& pv = vertices->front();
  size_t vtxIdx = 0;

  edm::Handle<reco::BeamSpot> beamSpot;
  if (useBeamSpot_)
    evt.getByToken(srcBeamSpot_, beamSpot);

  // Get transient track builder
  const TransientTrackBuilder& transTrackBuilder = es.getData(transTrackBuilderToken_);

  // Output collection
  auto outputVertices = std::make_unique<reco::VertexCollection>();
  outputVertices->reserve(1);

  // Create a new track collection for vertex refit
  std::vector<reco::TransientTrack> transTracks;

  // loop over the PFCandidates
  for (const auto& cand : (*cands)) {
    if (cand.charge() == 0 || cand.vertexRef().isNull())
      continue;
    if (cand.bestTrack() == nullptr)
      continue;
    auto key = cand.vertexRef().key();
    auto quality = cand.pvAssociationQuality();
    if (key != vtxIdx ||
        (quality != pat::PackedCandidate::UsedInFitTight && quality != pat::PackedCandidate::UsedInFitLoose))
      continue;
    if (useEleKfTracks_ && std::abs(cand.pdgId()) == 11)
      continue;
    transTracks.push_back(transTrackBuilder.build(cand.bestTrack()));
  }

  // loop over the lostTracks
  if (useLostTracks_) {
    for (const auto& cand : (*lostTrackCands)) {
      if (cand.charge() == 0 || cand.vertexRef().isNull())
        continue;
      if (cand.bestTrack() == nullptr)
        continue;
      auto key = cand.vertexRef().key();
      auto quality = cand.pvAssociationQuality();
      if (key != vtxIdx ||
          (quality != pat::PackedCandidate::UsedInFitTight && quality != pat::PackedCandidate::UsedInFitLoose))
        continue;
      transTracks.push_back(transTrackBuilder.build(cand.bestTrack()));
    }
  }

  // loop over the electronKfTracks
  if (useEleKfTracks_) {
    for (const auto& cand : (*eleKfTrackCands)) {
      if (cand.charge() == 0 || cand.vertexRef().isNull())
        continue;
      if (cand.bestTrack() == nullptr)
        continue;
      auto key = cand.vertexRef().key();
      auto quality = cand.pvAssociationQuality();
      if (key != vtxIdx ||
          (quality != pat::PackedCandidate::UsedInFitTight && quality != pat::PackedCandidate::UsedInFitLoose))
        continue;
      transTracks.push_back(transTrackBuilder.build(cand.bestTrack()));
    }
  }

  // Refit the vertex
  TransientVertex transVtx;
  reco::Vertex refitPV(pv);  // initialized to the original PV

  bool fitOK = true;
  if (transTracks.size() >= 3) {
    AdaptiveVertexFitter avf;
    avf.setWeightThreshold(0.1);  // weight per track, allow almost every fit, else --> exception
    if (!useBeamSpot_) {
      transVtx = avf.vertex(transTracks);
    } else {
      transVtx = avf.vertex(transTracks, *beamSpot);
    }
    if (!transVtx.isValid()) {
      fitOK = false;
    } else {
      //MB: protect against rare cases when transVtx is valid but its postion is ill-defined
      if (!std::isfinite(transVtx.position().z()))  //MB: it is enough to check one coordinate (?)
        fitOK = false;
    }
  } else
    fitOK = false;
  if (fitOK) {
    refitPV = transVtx;
  }

  outputVertices->push_back(refitPV);

  evt.put(std::move(outputVertices));
}

void PATRefitVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // patRefitVertexProducer
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("srcVertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("srcCands", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("srcLostTracks", edm::InputTag("lostTracks"));
  desc.add<edm::InputTag>("srcEleKfTracks", edm::InputTag("lostTracks:eleTracks"));
  desc.add<edm::InputTag>("srcBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<bool>("useBeamSpot", true)->setComment("Refit PV with beam-spot constraint");
  desc.add<bool>("useLostTracks", true)
      ->setComment("Use collection of tracks not used by PF-candidates, aka lost-tracks");
  desc.add<bool>("useEleKfTracks", true)
      ->setComment("Use collection of electron KF-tracks instead of GSF-tracks of electron PF-candidates");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATRefitVertexProducer);
