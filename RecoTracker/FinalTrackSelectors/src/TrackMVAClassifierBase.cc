#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <cassert>

void TrackMVAClassifierBase::fill(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("src", edm::InputTag());
  desc.add<edm::InputTag>("beamspot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("firstStepPrimaryVertices"));
  desc.add<bool>("ignoreVertices", false);
  // default cuts for "cut based classification"
  std::vector<double> cuts = {-.7, 0.1, .7};
  desc.add<std::vector<double>>("qualityCuts", cuts);
}

TrackMVAClassifierBase::~TrackMVAClassifierBase() {}

TrackMVAClassifierBase::TrackMVAClassifierBase(const edm::ParameterSet& cfg)
    : src_(consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("src"))),
      beamspot_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamspot"))),
      vertices_(mayConsume<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
      ignoreVertices_(cfg.getParameter<bool>("ignoreVertices")) {
  auto const& qv = cfg.getParameter<std::vector<double>>("qualityCuts");
  assert(qv.size() == 3);
  std::copy(std::begin(qv), std::end(qv), std::begin(qualityCuts));

  produces<MVACollection>("MVAValues");
  produces<QualityMaskCollection>("QualityMasks");
}

void TrackMVAClassifierBase::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get tracks
  edm::Handle<reco::TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack);
  auto const& tracks(*hSrcTrack);

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);

  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  evt.getByToken(vertices_, hVtx);

  initEvent(es);

  // products
  auto mvaPairs = std::make_unique<MVAPairCollection>(tracks.size(), std::make_pair(-99.f, true));
  auto mvas = std::make_unique<MVACollection>(tracks.size(), -99.f);
  auto quals = std::make_unique<QualityMaskCollection>(tracks.size(), 0);

  if (hVtx.isValid() && !ignoreVertices_) {
    computeMVA(tracks, *hBsp, *hVtx, *mvaPairs);
  } else {
    if (!ignoreVertices_)
      edm::LogWarning("TrackMVAClassifierBase")
          << "ignoreVertices is set to False in the configuration, but the vertex collection is not valid";
    std::vector<reco::Vertex> vertices;
    computeMVA(tracks, *hBsp, vertices, *mvaPairs);
  }
  assert((*mvaPairs).size() == tracks.size());

  unsigned int k = 0;
  for (auto const& output : *mvaPairs) {
    if (output.second) {
      (*mvas)[k] = output.first;
    } else {
      // If the MVA value is known to be unreliable, force into generalTracks collection
      (*mvas)[k] = std::max(output.first, float(qualityCuts[0] + 0.001));
    }
    float mva = (*mvas)[k];
    (*quals)[k++] = (mva > qualityCuts[0]) << reco::TrackBase::loose |
                    (mva > qualityCuts[1]) << reco::TrackBase::tight |
                    (mva > qualityCuts[2]) << reco::TrackBase::highPurity;
  }

  evt.put(std::move(mvas), "MVAValues");
  evt.put(std::move(quals), "QualityMasks");
}
