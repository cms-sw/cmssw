#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/TPS.h"

using namespace Phase2L1GMT;

TPS::TPS(const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getParameter<int>("verbose")),
      tt_track_converter_(new TrackConverter(iConfig.getParameter<edm::ParameterSet>("trackConverter"))),
      tps_(new TPSAlgorithm(iConfig.getParameter<edm::ParameterSet>("trackMatching"))),
      isolation_(new Isolation(iConfig.getParameter<edm::ParameterSet>("isolation"))) {}

TPS::~TPS() {}

std::vector<l1t::TrackerMuon> TPS::processEvent(const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks,
                                                const l1t::MuonStubRefVector& muonStubs) {
  //Split tracks to the links as they come
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks0 = associateTracksWithNonant(tracks, 0);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks1 = associateTracksWithNonant(tracks, 1);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks2 = associateTracksWithNonant(tracks, 2);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks3 = associateTracksWithNonant(tracks, 3);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks4 = associateTracksWithNonant(tracks, 4);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks5 = associateTracksWithNonant(tracks, 5);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks6 = associateTracksWithNonant(tracks, 6);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks7 = associateTracksWithNonant(tracks, 7);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks8 = associateTracksWithNonant(tracks, 8);

  //Convert TT tracks to our internal tracking format
  std::vector<ConvertedTTTrack> convertedTracks0 = tt_track_converter_->convertTracks(tracks0);
  std::vector<ConvertedTTTrack> convertedTracks1 = tt_track_converter_->convertTracks(tracks1);
  std::vector<ConvertedTTTrack> convertedTracks2 = tt_track_converter_->convertTracks(tracks2);
  std::vector<ConvertedTTTrack> convertedTracks3 = tt_track_converter_->convertTracks(tracks3);
  std::vector<ConvertedTTTrack> convertedTracks4 = tt_track_converter_->convertTracks(tracks4);
  std::vector<ConvertedTTTrack> convertedTracks5 = tt_track_converter_->convertTracks(tracks5);
  std::vector<ConvertedTTTrack> convertedTracks6 = tt_track_converter_->convertTracks(tracks6);
  std::vector<ConvertedTTTrack> convertedTracks7 = tt_track_converter_->convertTracks(tracks7);
  std::vector<ConvertedTTTrack> convertedTracks8 = tt_track_converter_->convertTracks(tracks8);

  //Transition stubs to different nonants with overlap
  l1t::MuonStubRefVector stubs0 = associateStubsWithNonant(muonStubs, 0);
  l1t::MuonStubRefVector stubs1 = associateStubsWithNonant(muonStubs, 1);
  l1t::MuonStubRefVector stubs2 = associateStubsWithNonant(muonStubs, 2);
  l1t::MuonStubRefVector stubs3 = associateStubsWithNonant(muonStubs, 3);
  l1t::MuonStubRefVector stubs4 = associateStubsWithNonant(muonStubs, 4);
  l1t::MuonStubRefVector stubs5 = associateStubsWithNonant(muonStubs, 5);
  l1t::MuonStubRefVector stubs6 = associateStubsWithNonant(muonStubs, 6);
  l1t::MuonStubRefVector stubs7 = associateStubsWithNonant(muonStubs, 7);
  l1t::MuonStubRefVector stubs8 = associateStubsWithNonant(muonStubs, 8);

  //run track - muon matching per nonant
  std::vector<PreTrackMatchedMuon> mu0 = tps_->processNonant(convertedTracks0, stubs0);
  std::vector<PreTrackMatchedMuon> mu1 = tps_->processNonant(convertedTracks1, stubs1);
  std::vector<PreTrackMatchedMuon> mu2 = tps_->processNonant(convertedTracks2, stubs2);
  std::vector<PreTrackMatchedMuon> mu3 = tps_->processNonant(convertedTracks3, stubs3);
  std::vector<PreTrackMatchedMuon> mu4 = tps_->processNonant(convertedTracks4, stubs4);
  std::vector<PreTrackMatchedMuon> mu5 = tps_->processNonant(convertedTracks5, stubs5);
  std::vector<PreTrackMatchedMuon> mu6 = tps_->processNonant(convertedTracks6, stubs6);
  std::vector<PreTrackMatchedMuon> mu7 = tps_->processNonant(convertedTracks7, stubs7);
  std::vector<PreTrackMatchedMuon> mu8 = tps_->processNonant(convertedTracks8, stubs8);
  //clean neighboring nonants
  std::vector<PreTrackMatchedMuon> muCleaned = tps_->cleanNeighbor(mu0, mu8, mu1, true);
  std::vector<PreTrackMatchedMuon> muCleaned1 = tps_->cleanNeighbor(mu1, mu0, mu2, false);
  std::vector<PreTrackMatchedMuon> muCleaned2 = tps_->cleanNeighbor(mu2, mu1, mu3, true);
  std::vector<PreTrackMatchedMuon> muCleaned3 = tps_->cleanNeighbor(mu3, mu2, mu4, false);
  std::vector<PreTrackMatchedMuon> muCleaned4 = tps_->cleanNeighbor(mu4, mu3, mu5, true);
  std::vector<PreTrackMatchedMuon> muCleaned5 = tps_->cleanNeighbor(mu5, mu4, mu6, false);
  std::vector<PreTrackMatchedMuon> muCleaned6 = tps_->cleanNeighbor(mu6, mu5, mu7, true);
  std::vector<PreTrackMatchedMuon> muCleaned7 = tps_->cleanNeighbor(mu7, mu6, mu8, false);
  std::vector<PreTrackMatchedMuon> muCleaned8 =
      tps_->cleanNeighbor(mu8, mu7, mu0, false);  //ARGH! 9 sectors - so some duplicates very rarely

  //merge all the collections
  std::copy(muCleaned1.begin(), muCleaned1.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned2.begin(), muCleaned2.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned3.begin(), muCleaned3.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned4.begin(), muCleaned4.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned5.begin(), muCleaned5.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned6.begin(), muCleaned6.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned7.begin(), muCleaned7.end(), std::back_inserter(muCleaned));
  std::copy(muCleaned8.begin(), muCleaned8.end(), std::back_inserter(muCleaned));

  std::vector<l1t::TrackerMuon> trackMatchedMuonsNoIso = tps_->convert(muCleaned, 32);

  //Isolation and tau3mu will read those muons and all 9 collections of convertedTracks*
  std::vector<ConvertedTTTrack> convertedTracks = convertedTracks0;
  std::copy(convertedTracks1.begin(), convertedTracks1.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks2.begin(), convertedTracks2.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks3.begin(), convertedTracks3.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks4.begin(), convertedTracks4.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks5.begin(), convertedTracks5.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks6.begin(), convertedTracks6.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks7.begin(), convertedTracks7.end(), std::back_inserter(convertedTracks));
  std::copy(convertedTracks8.begin(), convertedTracks8.end(), std::back_inserter(convertedTracks));

  //sorter here:
  std::vector<l1t::TrackerMuon> sortedTrackMuonsNoIso = tps_->sort(trackMatchedMuonsNoIso, 12);

  tps_->SetQualityBits(sortedTrackMuonsNoIso);

  isolation_->isolation_allmu_alltrk(sortedTrackMuonsNoIso, convertedTracks);

  //tauto3mu_->GetTau3Mu(sortedTrackMuonsNoIso, convertedTracks);

  tps_->outputGT(sortedTrackMuonsNoIso);

  return sortedTrackMuonsNoIso;  //when we add more collections like tau3mu etc we change that
}

std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > TPS::associateTracksWithNonant(
    const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks, uint processor) {
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > out;
  for (const auto& track : tracks) {
    if (track->phiSector() == processor) {
      out.push_back(track);
    }
  }
  return out;
}

l1t::SAMuonRefVector TPS::associateMuonsWithNonant(const l1t::SAMuonRefVector& muons, uint processor) {
  l1t::SAMuonRefVector out;

  ap_int<BITSPHI> center = ap_int<BITSPHI>(processor * 910);

  for (const auto& s : muons) {
    ap_int<BITSSTUBCOORD> deltaPhi = s->hwPhi() - center;
    ap_uint<BITSPHI - 1> absDeltaPhi =
        (deltaPhi < 0) ? ap_uint<BITSPHI - 1>(-deltaPhi) : ap_uint<BITSPHI - 1>(deltaPhi);
    if (absDeltaPhi < 683)
      out.push_back(s);
  }
  return out;
}

l1t::MuonStubRefVector TPS::associateStubsWithNonant(const l1t::MuonStubRefVector& allStubs, uint processor) {
  l1t::MuonStubRefVector out;

  ap_int<BITSSTUBCOORD> center = ap_int<BITSSTUBCOORD>((processor * 910) / 8);  //was 32

  for (const auto& s : allStubs) {
    ap_int<BITSSTUBCOORD> phi = 0;
    if (s->quality() & 0x1)
      phi = s->coord1();
    else
      phi = s->coord2();

    ap_int<BITSSTUBCOORD> deltaPhi = phi - center;
    ap_uint<BITSSTUBCOORD - 1> absDeltaPhi =
        (deltaPhi < 0) ? ap_uint<BITSSTUBCOORD - 1>(-deltaPhi) : ap_uint<BITSSTUBCOORD - 1>(deltaPhi);
    if (absDeltaPhi < 168)  //was 42
      out.push_back(s);
  }
  return out;
}
