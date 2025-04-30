/////////////////////////// BPHTrackMerger ////////////////////////////////
/// original authors: G Karathanasis (CERN),  G Melachroinos (NKUA)
// Takes Lost tracks and packed candidates filters them removes overlap and
// appl// -ies dz cut wrt to a dilepton vertex. Also applies selection cuts

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "helper.h"

class BPHTrackMerger : public edm::global::EDProducer<> {
 public:
  // would it be useful to give this a bit more standard structure?
  explicit BPHTrackMerger(const edm::ParameterSet &cfg)
      : bFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        beamSpotSrc_(consumes<reco::BeamSpot>(
            cfg.getParameter<edm::InputTag>("beamSpot"))),
        tracksToken_(consumes<pat::PackedCandidateCollection>(
            cfg.getParameter<edm::InputTag>("tracks"))),
        lostTracksToken_(consumes<pat::PackedCandidateCollection>(
            cfg.getParameter<edm::InputTag>("lostTracks"))),
        dileptonToken_(consumes<pat::CompositeCandidateCollection>(
            cfg.getParameter<edm::InputTag>("dileptons"))),
        muonToken_(consumes<pat::MuonCollection>(
            cfg.getParameter<edm::InputTag>("muons"))),
        eleToken_(consumes<pat::ElectronCollection>(
            cfg.getParameter<edm::InputTag>("electrons"))),
        pvToken_(consumes<std::vector<reco::Vertex>>(
            cfg.getParameter<edm::InputTag>("pvSrc"))),
        maxDzDilep_(cfg.getParameter<double>("maxDzDilep")),
        dcaSig_(cfg.getParameter<double>("dcaSig")),
        track_selection_(cfg.getParameter<std::string>("trackSelection")) {
    produces<pat::CompositeCandidateCollection>("SelectedTracks");
    produces<TransientTrackCollection>("SelectedTransientTracks");
    produces<edm::Association<pat::CompositeCandidateCollection>>(
        "SelectedTracks");
  }

  ~BPHTrackMerger() override {}

  void produce(edm::StreamID, edm::Event &,
               const edm::EventSetup &) const override;

 private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotSrc_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> tracksToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> lostTracksToken_;
  const edm::EDGetTokenT<pat::CompositeCandidateCollection> dileptonToken_;
  const edm::EDGetTokenT<pat::MuonCollection> muonToken_;
  const edm::EDGetTokenT<pat::ElectronCollection> eleToken_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;

  // selections
  const double maxDzDilep_;
  const double dcaSig_;
  const StringCutObjectSelector<pat::PackedCandidate> track_selection_;
};

void BPHTrackMerger::produce(edm::StreamID, edm::Event &evt,
                             edm::EventSetup const &stp) const {
  // input
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  evt.getByToken(beamSpotSrc_, beamSpotHandle);
  if (!beamSpotHandle.isValid()) {
    edm::LogError("BToKstllProducer") << "No beam spot available from Event";
  }
  const reco::BeamSpot &beamSpot = *beamSpotHandle;

  const auto &bField = stp.getData(bFieldToken_);
  edm::Handle<pat::PackedCandidateCollection> tracks;
  evt.getByToken(tracksToken_, tracks);
  edm::Handle<pat::PackedCandidateCollection> lostTracks;
  evt.getByToken(lostTracksToken_, lostTracks);
  edm::Handle<pat::CompositeCandidateCollection> dileptons;
  evt.getByToken(dileptonToken_, dileptons);

  edm::Handle<pat::MuonCollection> muons;
  evt.getByToken(muonToken_, muons);
  edm::Handle<pat::ElectronCollection> pfele;
  evt.getByToken(eleToken_, pfele);

  // for lost tracks / pf discrimination
  unsigned int nTracks = tracks->size();
  unsigned int totalTracks = nTracks + lostTracks->size();

  // ok this was CompositeCandidateCollection
  std::unique_ptr<pat::CompositeCandidateCollection> tracks_out(
      new pat::CompositeCandidateCollection);
  std::unique_ptr<TransientTrackCollection> trans_tracks_out(
      new TransientTrackCollection);

  std::vector<std::pair<pat::CompositeCandidate, reco::TransientTrack>>
      vectrk_ttrk;
  // try topreserve same logic avoiding the copy of the full collection
  /*
  //correct logic but a bit convoluted -> changing to smthn simpler
   std::vector<pat::PackedCandidate> totalTracks(*tracks);
   totalTracks.insert(totalTracks.end(),lostTracks->begin(),lostTracks->end());
  */

  // Retrieve the primary vertex collection
  edm::Handle<std::vector<reco::Vertex>> pvs;
  evt.getByToken(pvToken_, pvs);

  std::vector<int> match_indices(totalTracks, -1);
  // for loop is better to be range based - especially for large ensembles
  for (unsigned int iTrk = 0; iTrk < totalTracks; ++iTrk) {
    const pat::PackedCandidate &trk =
        (iTrk < nTracks) ? (*tracks)[iTrk] : (*lostTracks)[iTrk - nTracks];

    // arranging cuts for speed
    if (!trk.hasTrackDetails()) continue;
    if (fabs(trk.pdgId()) != 211) continue;  // do we want also to keep muons?
    if (!track_selection_(trk)) continue;

    bool skipTrack = true;
    float dzTrg = 0.0;
    for (const pat::CompositeCandidate &dilep : *dileptons) {
      // if dz is negative it is deactivated
      if (fabs(trk.vz() - dilep.vz()) > maxDzDilep_ && maxDzDilep_ > 0)
        continue;
      skipTrack = false;
      dzTrg = trk.vz() - dilep.vz();
      break;  // at least for one dilepton candidate to pass this cuts
    }

    // if track is far from all dilepton candidate
    if (skipTrack) continue;

    // high purity requirment applied only in packedCands
    if (iTrk < nTracks && !trk.trackHighPurity()) continue;
    const reco::TransientTrack trackTT((*trk.bestTrack()), &bField);

    // distance closest approach in x,y wrt beam spot
    std::pair<double, double> DCA = bph::computeDCA(trackTT, beamSpot);
    float DCABS = DCA.first;
    float DCABSErr = DCA.second;
    float DCASig = (DCABSErr != 0 && float(DCABSErr) == DCABSErr)
                       ? fabs(DCABS / DCABSErr)
                       : -1;

    if (DCASig > dcaSig_ && dcaSig_ > 0) continue;

    // clean tracks wrt to all muons
    int matchedToMuon = 0;
    for (const pat::Muon &imutmp : *muons) {
      for (unsigned int i = 0; i < imutmp.numberOfSourceCandidatePtrs(); ++i) {
        if (!((imutmp.sourceCandidatePtr(i)).isNonnull() &&
              (imutmp.sourceCandidatePtr(i)).isAvailable()))
          continue;

        const edm::Ptr<reco::Candidate> &source = imutmp.sourceCandidatePtr(i);
        if (source.id() == tracks.id() && source.key() == iTrk) {
          matchedToMuon = 1;
          break;
        }
      }
    }

    // clean tracks wrt to all pf electrons
    int matchedToEle = 0;
    for (const pat::Electron &ietmp : *pfele) {
      for (unsigned int i = 0; i < ietmp.numberOfSourceCandidatePtrs(); ++i) {
        if (!((ietmp.sourceCandidatePtr(i)).isNonnull() &&
              (ietmp.sourceCandidatePtr(i)).isAvailable()))
          continue;
        const edm::Ptr<reco::Candidate> &source = ietmp.sourceCandidatePtr(i);
        if (source.id() == tracks.id() && source.key() == iTrk) {
          matchedToEle = 1;
          break;
        }
      }
    }

    // IP
    const reco::Vertex &pv0 = pvs->front();

    // output
    pat::CompositeCandidate pcand;
    pcand.setP4(trk.p4());
    pcand.setCharge(trk.charge());
    pcand.setVertex(trk.vertex());
    pcand.setPdgId(trk.pdgId());
    pcand.addUserInt("isPacked", (iTrk < nTracks));
    pcand.addUserInt("isLostTrk", (iTrk < nTracks) ? 0 : 1);
    pcand.addUserFloat("dxy", trk.dxy(pv0.position()));
    pcand.addUserFloat("dxyS", trk.dxy(pv0.position()) / trk.dxyError());
    pcand.addUserFloat("dz", trk.dz(pv0.position()));
    pcand.addUserFloat("dzS", trk.dz() / trk.dzError());
    pcand.addUserFloat("DCASig", DCASig);
    pcand.addUserFloat("dzTrg", dzTrg);
    pcand.addUserInt("isMatchedToMuon", matchedToMuon);
    pcand.addUserInt("isMatchedToEle", matchedToEle);
    pcand.addUserInt("nValidHits", trk.bestTrack()->found());
    pcand.addUserInt("keyPacked", iTrk);

    // Covariance matrix elements for helix parameters for decay time
    // uncertainty
    pcand.addUserFloat("covQopQop", trk.bestTrack()->covariance(0, 0));
    pcand.addUserFloat("covLamLam", trk.bestTrack()->covariance(1, 1));
    pcand.addUserFloat("covPhiPhi", trk.bestTrack()->covariance(2, 2));
    pcand.addUserFloat("covQopLam", trk.bestTrack()->covariance(0, 1));
    pcand.addUserFloat("covQopPhi", trk.bestTrack()->covariance(0, 2));
    pcand.addUserFloat("covLamPhi", trk.bestTrack()->covariance(1, 2));

    // Additional track parameters for tagging
    pcand.addUserFloat("ptErr", trk.bestTrack()->ptError());
    pcand.addUserFloat("normChi2", trk.bestTrack()->normalizedChi2());
    pcand.addUserInt("nValidPixelHits", trk.numberOfPixelHits());

    // adding the candidate in the composite stuff for fit (need to test)
    if (iTrk < nTracks)
      pcand.addUserCand("cand", edm::Ptr<pat::PackedCandidate>(tracks, iTrk));
    else
      pcand.addUserCand(
          "cand", edm::Ptr<pat::PackedCandidate>(lostTracks, iTrk - nTracks));

    // in order to avoid revoking the sxpensive ttrack builder many times and
    // still have everything sorted, we add them to vector of pairs
    match_indices[iTrk] = vectrk_ttrk.size();
    vectrk_ttrk.emplace_back(std::make_pair(pcand, trackTT));
  }

  std::vector<int> sort_indices(vectrk_ttrk.size());
  std::iota(sort_indices.begin(), sort_indices.end(), 0);

  // sort to be uniform with leptons
  // sort by index since we want to update the match too
  std::sort(sort_indices.begin(), sort_indices.end(),
            [&vectrk_ttrk](auto &iTrk1, auto &iTrk2) -> bool {
              return (vectrk_ttrk[iTrk1].first).pt() >
                     (vectrk_ttrk[iTrk2].first).pt();
            });
  // std::sort( vectrk_ttrk.begin(), vectrk_ttrk.end(),
  //            [] ( auto & trk1, auto & trk2) ->
  //            bool {return (trk1.first).pt() > (trk2.first).pt();}
  //          );

  // finally save ttrks and trks to the correct _out vectors
  // also fill the reverse matching
  std::vector<int> reverse_sort_indices(vectrk_ttrk.size());
  for (size_t iSort = 0; iSort < sort_indices.size(); iSort++) {
    auto iUnsortedTrack = sort_indices[iSort];
    auto &&trk = vectrk_ttrk[iUnsortedTrack];
    tracks_out->emplace_back(trk.first);
    trans_tracks_out->emplace_back(trk.second);
    reverse_sort_indices[iUnsortedTrack] = iSort;
  }

  // Now point the match indices to the sorted output collection
  std::transform(match_indices.begin(), match_indices.end(),
                 match_indices.begin(),
                 [&reverse_sort_indices](int iUnsortedTrack) {
                   if (iUnsortedTrack < 0) return -1;
                   return reverse_sort_indices[iUnsortedTrack];
                 });

  int unassoc = 0;
  for (auto iTrkAssoc : match_indices) {
    unassoc += iTrkAssoc < 0;
  }
  // std::clog << "There are " << unassoc << " unassociated tracks" <<
  // std::endl; std::clog << "Total tracks: " << totalTracks << " output tracks:
  // " <<  tracks_out->size() << std::endl;

  auto tracks_orphan_handle = evt.put(std::move(tracks_out), "SelectedTracks");
  evt.put(std::move(trans_tracks_out), "SelectedTransientTracks");

  // Associate PackedCandidates to the merged Track collection
  auto tracks_out_match =
      std::make_unique<edm::Association<pat::CompositeCandidateCollection>>(
          tracks_orphan_handle);
  edm::Association<pat::CompositeCandidateCollection>::Filler filler(
      *tracks_out_match);

  filler.insert(tracks, match_indices.begin(), match_indices.begin() + nTracks);
  filler.insert(lostTracks, match_indices.begin() + nTracks,
                match_indices.end());
  filler.fill();

  evt.put(std::move(tracks_out_match), "SelectedTracks");
}

// define this as a plug-in
DEFINE_FWK_MODULE(BPHTrackMerger);
