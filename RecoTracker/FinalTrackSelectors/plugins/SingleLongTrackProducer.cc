// user includes
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// ROOT includes
#include "TLorentzVector.h"

class SingleLongTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit SingleLongTrackProducer(const edm::ParameterSet &);
  ~SingleLongTrackProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muonsToken;
  const edm::EDGetTokenT<reco::VertexCollection> PrimVtxToken;

  const int minNumberOfLayers;
  const double matchInDr;
  const bool onlyValidHits;
  const bool debug;
  const double minPt;
  const double maxEta;
  const double maxDxy;
  const double maxDz;
};

SingleLongTrackProducer::SingleLongTrackProducer(const edm::ParameterSet &iConfig)
    : tracksToken{consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("allTracks"))},
      muonsToken{consumes<std::vector<reco::Muon>>(iConfig.getParameter<edm::InputTag>("matchMuons"))},
      PrimVtxToken{consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("PrimaryVertex"))},
      minNumberOfLayers{iConfig.getParameter<int>("minNumberOfLayers")},
      matchInDr{iConfig.getParameter<double>("requiredDr")},
      onlyValidHits{iConfig.getParameter<bool>("onlyValidHits")},
      debug{iConfig.getParameter<bool>("debug")},
      minPt{iConfig.getParameter<double>("minPt")},
      maxEta{iConfig.getParameter<double>("maxEta")},
      maxDxy{iConfig.getParameter<double>("maxDxy")},
      maxDz{iConfig.getParameter<double>("maxDz")} {
  produces<reco::TrackCollection>("").setBranchAlias("");
}

void SingleLongTrackProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  // register input collections:
  const auto &tracks = iEvent.getHandle(tracksToken);
  const auto &muons = iEvent.getHandle(muonsToken);

  const auto &vtx_h = iEvent.getHandle(PrimVtxToken);
  const reco::Vertex vtx = vtx_h->at(0);

  // register output collection:
  std::unique_ptr<reco::TrackCollection> goodTracks(new reco::TrackCollection);

  // Preselection of long quality tracks
  std::vector<reco::Track> selTracks;
  reco::Track bestTrack;
  unsigned int tMuon = 0;
  double fitProb = 100;

  for (const auto &track : *tracks) {
    const reco::HitPattern &hitpattern = track.hitPattern();
    double dRmin = 10;
    double chiNdof = track.normalizedChi2();
    double dxy = std::abs(track.dxy(vtx.position()));
    double dz = std::abs(track.dz(vtx.position()));

    if (track.pt() < minPt)
      continue;

    if (abs(track.eta()) > maxEta)
      continue;

    if (hitpattern.trackerLayersWithMeasurement() < minNumberOfLayers)
      continue;

    TLorentzVector pTrack(track.px(), track.py(), track.pz(), track.pt());

    // Long track needs to be close to a good muon
    for (const auto &m : *muons) {
      if (m.isTrackerMuon()) {
        tMuon++;
        reco::Track matchedTrack = *(m.innerTrack());
        TLorentzVector pMatched(matchedTrack.px(), matchedTrack.py(), matchedTrack.pz(), matchedTrack.pt());
        // match to general track in deltaR
        double dr = pTrack.DeltaR(pMatched);
        if (dr < dRmin)
          dRmin = dr;
      }
    }

    if (dRmin >= matchInDr)
      continue;
    // do vertex consistency:
    bool vertex_match = dxy < maxDxy && dz < maxDz;
    if (!(vertex_match))
      continue;
    if (track.validFraction() < 1.0)
      continue;
    // only save the track with the smallest chiNdof
    if (chiNdof < fitProb) {
      fitProb = chiNdof;
      bestTrack = track;
      bestTrack.setExtra(track.extra());
    }
    if (debug)
      edm::LogPrint("SingleLongTrackProducer") << " deltaR (general) track to matched Track: " << dRmin << std::endl;
    if (debug)
      edm::LogPrint("SingleLongTrackProducer") << "chi2Ndof:" << chiNdof << " best Track: " << fitProb << std::endl;
  }

  selTracks.push_back(bestTrack);

  if (debug)
    edm::LogPrint("SingleLongTrackProducer") << " number of Tracker Muons: " << tMuon << ", thereof "
                                             << selTracks.size() << " tracks passed preselection." << std::endl;

  // check hits validity in preselected tracks
  bool hitIsNotValid{false};

  for (const auto &track : selTracks) {
    reco::HitPattern hitpattern = track.hitPattern();
    int deref{0};

    // this checks track recHits
    try {  // (Un)Comment this line with /* to (not) allow for events with not valid hits
      auto hb = track.recHitsBegin();

      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        auto recHit = *(hb + h);
        auto const &hit = *recHit;

        if (onlyValidHits && !hit.isValid()) {
          hitIsNotValid = true;
          continue;
        }
      }
    } catch (cms::Exception const &e) {
      deref += 1;
      if (debug)
        std::cerr << e.explainSelf() << std::endl;
    }

    if (hitIsNotValid == true)
      break;  // (Un)Comment this line with */ to (not) allow for events with not valid hits

    int deref2{0};

    // this checks track hitPattern hits
    try {
      auto hb = track.recHitsBegin();

      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        uint32_t pHit = hitpattern.getHitPattern(reco::HitPattern::TRACK_HITS, h);

        auto recHit = *(hb + h);
        auto const &hit = *recHit;

        if (onlyValidHits && !hit.isValid()) {
          if (debug)
            edm::LogPrint("SingleLongTrackProducer") << "hit not valid: " << h << std::endl;
          continue;
        }

        // loop over the hits of the track.
        if (onlyValidHits && !(hitpattern.validHitFilter(pHit))) {
          if (debug)
            edm::LogPrint("SingleLongTrackProducer") << "hit not valid: " << h << std::endl;
          continue;
        }
      }
      goodTracks->push_back(track);
    } catch (cms::Exception const &e) {
      deref2 += 1;
      if (debug)
        std::cerr << e.explainSelf() << std::endl;
    }

    if (debug)
      edm::LogPrint("SingleLongTrackProducer")
          << "found tracks with " << deref << "missing valid hits and " << deref2 << " missing hit pattern";
  }

  // save track collection in event:
  iEvent.put(std::move(goodTracks), "");
}

void SingleLongTrackProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("allTracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("matchMuons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("PrimaryVertex", edm::InputTag("offlinePrimaryVertices"));
  desc.add<int>("minNumberOfLayers", 10);
  desc.add<double>("requiredDr", 0.01);
  desc.add<bool>("onlyValidHits", true);
  desc.add<bool>("debug", false);
  desc.add<double>("minPt", 15.0);
  desc.add<double>("maxEta", 2.2);
  desc.add<double>("maxDxy", 0.02);
  desc.add<double>("maxDz", 0.5);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SingleLongTrackProducer);
