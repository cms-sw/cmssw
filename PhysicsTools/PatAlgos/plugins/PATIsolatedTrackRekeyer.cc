#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include <vector>

class PATIsolatedTrackRekeyer : public edm::stream::EDProducer<> {
private:
  using IsoTracksC = std::vector<pat::IsolatedTrack>;
  using PackedCandsC = pat::PackedCandidateCollection;

  edm::EDGetTokenT<IsoTracksC> input_tracks_token_;
  edm::EDGetTokenT<PackedCandsC> packed_cands_token_;
  edm::EDGetTokenT<PackedCandsC> ori_packed_cands_token_;

public:
  PATIsolatedTrackRekeyer(edm::ParameterSet const& params)
      : input_tracks_token_{consumes<IsoTracksC>(params.getParameter<edm::InputTag>("src"))},
        packed_cands_token_{consumes<PackedCandsC>(params.getParameter<edm::InputTag>("packedCands"))},
        ori_packed_cands_token_{consumes<PackedCandsC>(params.getParameter<edm::InputTag>("packedCandsOri"))} {
    produces<IsoTracksC>();
  }

  ~PATIsolatedTrackRekeyer() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto out_tracks = std::make_unique<std::vector<pat::IsolatedTrack>>();

    edm::Handle<std::vector<pat::IsolatedTrack>> input_tracks;
    iEvent.getByToken(input_tracks_token_, input_tracks);

    edm::Handle<pat::PackedCandidateCollection> packed_cands;
    iEvent.getByToken(packed_cands_token_, packed_cands);

    edm::Handle<pat::PackedCandidateCollection> ori_packed_cands;
    iEvent.getByToken(ori_packed_cands_token_, ori_packed_cands);

    auto rekey = [&](pat::PackedCandidateRef const& ref) {
      return (ref.isNonnull() && ref.id() == ori_packed_cands.id())
                 ? pat::PackedCandidateRef(packed_cands, ref.key())
                 : ref;
    };

    for (const auto& track : *input_tracks) {
      // copy original pat object and append to vector

      auto const cand_ref = rekey(track.packedCandRef());
      auto const near_ref = rekey(track.nearestPFPackedCandRef());
      auto const lost_ref = rekey(track.nearestLostTrackPackedCandRef());

      out_tracks->emplace_back((pat::IsolatedTrack(track.pfIsolationDR03(),
                                                   track.miniPFIsolation(),
                                                   track.matchedCaloJetEmEnergy(),
                                                   track.matchedCaloJetHadEnergy(),
                                                   track.pfLepOverlap(),
                                                   track.pfNeutralSum(),
                                                   track.p4(),
                                                   track.charge(),
                                                   track.pdgId(),
                                                   track.dz(),
                                                   track.dxy(),
                                                   track.dzError(),
                                                   track.dxyError(),
                                                   track.hitPattern(),
                                                   track.dEdxStrip(),
                                                   track.dEdxPixel(),
                                                   track.fromPV(),
                                                   track.trackQuality(),
                                                   track.crossedEcalStatus(),
                                                   track.crossedHcalStatus(),
                                                   track.deltaEta(),
                                                   track.deltaPhi(),
                                                   cand_ref,
                                                   near_ref,
                                                   lost_ref)));
    }

    iEvent.put(std::move(out_tracks));
  }
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATIsolatedTrackRekeyer);
