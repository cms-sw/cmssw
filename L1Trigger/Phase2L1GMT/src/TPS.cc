#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/TPS.h"

using namespace Phase2L1GMT;

TPS::TPS(const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getParameter<int>("verbose")),
      tt_track_converter_(new TrackConverter(iConfig.getParameter<edm::ParameterSet>("trackConverter"))),
      tps_(new TPSAlgorithm(iConfig.getParameter<edm::ParameterSet>("trackMatching"))),
      isolation_(new Isolation(iConfig.getParameter<edm::ParameterSet>("isolation"))) {}

std::vector<l1t::TrackerMuon> TPS::processEvent(const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks,
                                                const l1t::MuonStubRefVector& muonStubs) {

  //Split tracks to the links as they come
  std::array<std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >, 9> loctracks;
  for (unsigned i = 0; i < 9; ++i)
    loctracks[i] = associateTracksWithNonant(tracks, i);

  //Convert TT tracks to our internal tracking format
  std::array<std::vector<ConvertedTTTrack>, 9> convertedTracks;
  for (unsigned i = 0; i < 9; ++i)
    convertedTracks[i] = tt_track_converter_->convertTracks(loctracks.at(i));

  //Transition stubs to different nonants with overlap
  std::array<l1t::MuonStubRefVector, 9> stubs;
  for (int i = 0; i < 9; ++i)
    stubs[i] = associateStubsWithNonant(muonStubs, i);
  
  //print all tracks and stubs for an event
  if (verbose_ == 1) {
    for (unsigned i = 0; i < 9; ++i) {
      edm::LogInfo("TPS") << "Sector " << i << ":\n"
	                  << "    L1 Tracks:";
      for (const auto& t : loctracks[i]) {
        edm::LogInfo("TPS") << "        RinvBits = " << ap_int<BITSTTCURV>(t->getRinvBits()) << ", eta = " << t->eta() << ", phi = " << t->phi() << std::flush;
        edm::LogInfo("TPS") << "            l1track word = " << std::setfill('0') << std::setw(8) << std::hex << ((t->getTrackWord())>>64).to_uint64() << std::setw(16) << (t->getTrackWord()).to_uint64() << std::dec << std::flush;
      }
      edm::LogInfo("TPS") << "    Converted Tracks:";
      for (const auto& track : convertedTracks[i]) {
	track.print("TPS", 8, false);
	track.printWord("TPS", 12, true);
      }
      edm::LogInfo("TPS") << "    Stubs:";
      for (const auto& stub : stubs[i]) {
	stub->printHybridStub("TPS", 8, false);
	stub->printHybridStubWord("TPS", 12, true);

      }
    }
  }
  
  //run track - muon matching per nonant
  std::array<std::vector<PreTrackMatchedMuon>, 9> mus;
  for (int i = 0; i < 9; ++i)
    mus[i] = tps_->processNonant(convertedTracks.at(i), stubs.at(i));

  //examine match and overall muon qualities pre and post cleaning
  if (verbose_ == 1) {
    edm::LogInfo("TPS") << "Before cleaning premuons:";
    for (unsigned i = 0; i < 9; ++i) {
      std::vector<PreTrackMatchedMuon> sector_mus = mus[i];
      for (const auto& sector_mu : sector_mus) {
	edm::LogInfo("TPS") << "    (Sector " << i << ")" << std::flush;
	sector_mu.print("TPS", 4, false);
        sector_mu.printWord("TPS", 8, true);	
      }
    }
  }
  
  //clean all of the nonants at once
  std::vector<PreTrackMatchedMuon> mergedPrecleaned;
  for (auto&& v : mus) {
    mergedPrecleaned.insert(mergedPrecleaned.end(), v.begin(), v.end());
  }
  std::vector<PreTrackMatchedMuon> mergedCleaned = tps_->cleanAll(mergedPrecleaned);
  
  if (verbose_ == 1) {
    edm::LogInfo("TPS") << "After cleaning premuons:";
    for (const auto& cleaned_mu : mergedCleaned) {
      cleaned_mu.print("TPS", 4, false);
      cleaned_mu.printWord("TPS", 8, true);
    }
  }

  std::vector<l1t::TrackerMuon> trackMatchedMuonsNoIso = tps_->convert(mergedCleaned, 32);

  //sorter here:
  std::vector<l1t::TrackerMuon> sortedTrackMuonsNoIso = tps_->sort(trackMatchedMuonsNoIso, 12);

  tps_->SetQualityBits(sortedTrackMuonsNoIso);

  //Isolation and tau3mu will read those muons and all 9 collections of convertedTracks*
  std::vector<ConvertedTTTrack> mergedconvertedTracks;
  for (auto&& v : convertedTracks) {
    mergedconvertedTracks.insert(mergedconvertedTracks.end(), v.begin(), v.end());
  }

  isolation_->isolation_allmu_alltrk(sortedTrackMuonsNoIso, mergedconvertedTracks);

  // To be enable when tau->3mu rate study is ready
  //tauto3mu_->GetTau3Mu(sortedTrackMuonsNoIso, mergedconvertedTracks);

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
    if (absDeltaPhi < 284) //was 284 //was 168 //was 42
      out.push_back(s);
  }
  return out;
}
