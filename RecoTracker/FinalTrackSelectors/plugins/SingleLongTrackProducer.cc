// user includes
#include "DataFormats/Math/interface/deltaR.h"
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

  // register output collection:
  std::unique_ptr<reco::TrackCollection> goodTracks(new reco::TrackCollection);

  // register input collections:
  const auto &tracks = iEvent.getHandle(tracksToken);
  if (!tracks.isValid()) {
    edm::LogError("SingleLongTrackProducer")
        << "Input track collection is not valid.\n Returning empty output track collection.";
    iEvent.put(std::move(goodTracks), "");
    return;
  }

  const auto &muons = iEvent.getHandle(muonsToken);
  if (!muons.isValid() && matchInDr > 0.) {
    edm::LogError("SingleLongTrackProducer")
        << "Input muon collection is not valid.\n Returning empty output track collection.";
    iEvent.put(std::move(goodTracks), "");
    return;
  }

  const auto &vertices = iEvent.getHandle(PrimVtxToken);
  if (!vertices.isValid()) {
    edm::LogError("SingleLongTrackProducer")
        << "Input vertex collection is not valid.\n Returning empty output track collection.";
    iEvent.put(std::move(goodTracks), "");
    return;
  }

  // Preselection of long quality tracks
  const reco::Vertex &vtx = vertices->at(0);
  std::vector<reco::Track> selTracks;
  reco::Track bestTrack;
  unsigned int tMuon = 0;
  double fitProb = std::numeric_limits<double>::max();

  for (const auto &track : *tracks) {
    const reco::HitPattern &hitpattern = track.hitPattern();
    double dR2min = std::numeric_limits<double>::max();
    double chiNdof = track.normalizedChi2();
    double dxy = std::abs(track.dxy(vtx.position()));
    double dz = std::abs(track.dz(vtx.position()));

    if (track.pt() < minPt || std::abs(track.eta()) > maxEta ||
        hitpattern.trackerLayersWithMeasurement() < minNumberOfLayers) {
      continue;
    }

    if (matchInDr > 0.0) {
      for (const auto &muon : *muons) {
        if (muon.isTrackerMuon()) {
          tMuon++;
          double dr2 = reco::deltaR2(track, *(muon.innerTrack()));
          dR2min = std::min(dR2min, dr2);
        }
      }
      if (dR2min >= matchInDr * matchInDr)
        continue;
    }

    if (dxy < maxDxy && dz < maxDz && track.validFraction() >= 1.0 && chiNdof < fitProb) {
      fitProb = chiNdof;
      bestTrack = track;
    }

    if (debug) {
      edm::LogPrint("SingleLongTrackProducer") << "deltaR2 (general) track to matched Track: " << dR2min
                                               << " chi2Ndof: " << chiNdof << " best Track chi2Ndof: " << fitProb;
    }
  }

  // Only push if bestTrack was set
  if (bestTrack.recHitsOk()) {
    selTracks.push_back(bestTrack);
  }

  if (debug)
    edm::LogPrint("SingleLongTrackProducer")
        << " number of Tracker Muons: " << tMuon << ", thereof " << selTracks.size() << " tracks passed preselection.";

  // check hits validity in preselected tracks
  bool hitIsNotValid{false};

  for (const auto &track : selTracks) {
    // Check validity of track extra and rechits
    if (track.recHitsOk()) {
      reco::HitPattern hitpattern = track.hitPattern();
      auto hb = track.recHitsBegin();

      // Checking if rechits are valid
      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        auto recHit = *(hb + h);
        auto const &hit = *recHit;
        if (onlyValidHits && !hit.isValid()) {
          hitIsNotValid = true;
          continue;
        }
      }

      if (hitIsNotValid == true)
        break;  // (Un)Comment this line to (not) allow for events with not valid hits

      // Checking if hitpattern hits are valid
      for (unsigned int h = 0; h < track.recHitsSize(); h++) {
        uint32_t pHit = hitpattern.getHitPattern(reco::HitPattern::TRACK_HITS, h);
        if (onlyValidHits && !(hitpattern.validHitFilter(pHit))) {
          if (debug)
            edm::LogPrint("SingleLongTrackProducer") << "hit not valid: " << h;
          continue;
        }
      }
      goodTracks->push_back(track);
    } else
      break;
  }

  if (debug) {
    auto const &moduleType = moduleDescription().moduleName();
    auto const &moduleLabel = moduleDescription().moduleLabel();
    edm::LogPrint("SingleLongTrackProducer") << "[" << moduleType << "] (" << moduleLabel << ") "
                                             << " output track size: " << goodTracks.get()->size();
  }

  // save track collection in event:
  iEvent.put(std::move(goodTracks), "");
}

void SingleLongTrackProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("allTracks", edm::InputTag("generalTracks"))->setComment("input track collection");
  desc.add<edm::InputTag>("matchMuons", edm::InputTag("earlyMuons"))->setComment("input muon collection for matching");
  desc.add<edm::InputTag>("PrimaryVertex", edm::InputTag("offlinePrimaryVertices"))
      ->setComment("input primary vertex collection");
  desc.add<int>("minNumberOfLayers", 10)->setComment("minimum number of layers");
  desc.add<double>("requiredDr", 0.01)->setComment("matching muons deltaR. If negative do not match");
  desc.add<bool>("onlyValidHits", true)->setComment("use only valid hits");
  desc.add<bool>("debug", false)->setComment("verbose?");
  desc.add<double>("minPt", 15.0)->setComment("minimum pT");
  desc.add<double>("maxEta", 2.2)->setComment("maximum pseudorapidity (absolute value)");
  desc.add<double>("maxDxy", 0.02)->setComment("maximum transverse impact parameter");
  desc.add<double>("maxDz", 0.5)->setComment("maximum longitudinal impact parameter");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SingleLongTrackProducer);
