#ifndef PHASE2GMT_NODE
#define PHASE2GMT_NODE
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackConverter.h"
#include "ROITempAssociator.h"
#include "TrackMuonMatchAlgorithm.h"
#include "Isolation.h"
#include "Tauto3Mu.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

namespace Phase2L1GMT {

  class Node {
  public:
    Node(const edm::ParameterSet& iConfig)
        : verbose_(iConfig.getParameter<int>("verbose")),
          tt_track_converter_(new TrackConverter(iConfig.getParameter<edm::ParameterSet>("trackConverter"))),
          roi_assoc_(new ROITempAssociator(iConfig.getParameter<edm::ParameterSet>("roiTrackAssociator"))),
          track_mu_match_(new TrackMuonMatchAlgorithm(iConfig.getParameter<edm::ParameterSet>("trackMatching"))),
          isolation_(new Isolation(iConfig.getParameter<edm::ParameterSet>("isolation"))),
          tauto3mu_(new Tauto3Mu(iConfig.getParameter<edm::ParameterSet>("tauto3mu"))) {}

    ~Node() {}

    std::vector<l1t::TrackerMuon> processEvent(const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks,
                                               const l1t::ObjectRefBxCollection<l1t::RegionalMuonCand>& muonTracks,
                                               const l1t::MuonStubRefVector& stubs) {
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

      //Transition stubs to different nonants with overlap
      l1t::MuonStubRefVector stubs0 = associateStubsWithNonant(stubs, 0);
      l1t::MuonStubRefVector stubs1 = associateStubsWithNonant(stubs, 1);
      l1t::MuonStubRefVector stubs2 = associateStubsWithNonant(stubs, 2);
      l1t::MuonStubRefVector stubs3 = associateStubsWithNonant(stubs, 3);
      l1t::MuonStubRefVector stubs4 = associateStubsWithNonant(stubs, 4);
      l1t::MuonStubRefVector stubs5 = associateStubsWithNonant(stubs, 5);
      l1t::MuonStubRefVector stubs6 = associateStubsWithNonant(stubs, 6);
      l1t::MuonStubRefVector stubs7 = associateStubsWithNonant(stubs, 7);
      l1t::MuonStubRefVector stubs8 = associateStubsWithNonant(stubs, 8);

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

      //Build ROIs per nonant
      std::vector<MuonROI> rois0 = roi_assoc_->associate(0, muonTracks, stubs0);
      std::vector<MuonROI> rois1 = roi_assoc_->associate(0, muonTracks, stubs1);
      std::vector<MuonROI> rois2 = roi_assoc_->associate(0, muonTracks, stubs2);
      std::vector<MuonROI> rois3 = roi_assoc_->associate(0, muonTracks, stubs3);
      std::vector<MuonROI> rois4 = roi_assoc_->associate(0, muonTracks, stubs4);
      std::vector<MuonROI> rois5 = roi_assoc_->associate(0, muonTracks, stubs5);
      std::vector<MuonROI> rois6 = roi_assoc_->associate(0, muonTracks, stubs6);
      std::vector<MuonROI> rois7 = roi_assoc_->associate(0, muonTracks, stubs7);
      std::vector<MuonROI> rois8 = roi_assoc_->associate(0, muonTracks, stubs8);

      //run track - muon matching per nonant
      std::vector<PreTrackMatchedMuon> mu0 = track_mu_match_->processNonant(convertedTracks0, rois0);
      std::vector<PreTrackMatchedMuon> mu1 = track_mu_match_->processNonant(convertedTracks1, rois1);
      std::vector<PreTrackMatchedMuon> mu2 = track_mu_match_->processNonant(convertedTracks2, rois2);
      std::vector<PreTrackMatchedMuon> mu3 = track_mu_match_->processNonant(convertedTracks3, rois3);
      std::vector<PreTrackMatchedMuon> mu4 = track_mu_match_->processNonant(convertedTracks4, rois4);
      std::vector<PreTrackMatchedMuon> mu5 = track_mu_match_->processNonant(convertedTracks5, rois5);
      std::vector<PreTrackMatchedMuon> mu6 = track_mu_match_->processNonant(convertedTracks6, rois6);
      std::vector<PreTrackMatchedMuon> mu7 = track_mu_match_->processNonant(convertedTracks7, rois7);
      std::vector<PreTrackMatchedMuon> mu8 = track_mu_match_->processNonant(convertedTracks8, rois8);
      if (verbose_)
        printf("Matching Nonant 5 with %zu tracks and %zu rois and %zu stubs\n",
               convertedTracks5.size(),
               rois5.size(),
               stubs5.size());

      //clean neighboring nonants
      std::vector<PreTrackMatchedMuon> muCleaned = track_mu_match_->cleanNeighbor(mu0, mu8, mu1, true);
      std::vector<PreTrackMatchedMuon> muCleaned1 = track_mu_match_->cleanNeighbor(mu1, mu0, mu2, false);
      std::vector<PreTrackMatchedMuon> muCleaned2 = track_mu_match_->cleanNeighbor(mu2, mu1, mu3, true);
      std::vector<PreTrackMatchedMuon> muCleaned3 = track_mu_match_->cleanNeighbor(mu3, mu2, mu4, false);
      std::vector<PreTrackMatchedMuon> muCleaned4 = track_mu_match_->cleanNeighbor(mu4, mu3, mu5, true);
      std::vector<PreTrackMatchedMuon> muCleaned5 = track_mu_match_->cleanNeighbor(mu5, mu4, mu6, false);
      std::vector<PreTrackMatchedMuon> muCleaned6 = track_mu_match_->cleanNeighbor(mu6, mu5, mu7, true);
      std::vector<PreTrackMatchedMuon> muCleaned7 = track_mu_match_->cleanNeighbor(mu7, mu6, mu8, false);
      std::vector<PreTrackMatchedMuon> muCleaned8 =
          track_mu_match_->cleanNeighbor(mu8, mu7, mu0, false);  //ARGH! 9 sectors - so some duplicates very rarely

      //merge all the collections
      std::copy(muCleaned1.begin(), muCleaned1.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned2.begin(), muCleaned2.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned3.begin(), muCleaned3.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned4.begin(), muCleaned4.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned5.begin(), muCleaned5.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned6.begin(), muCleaned6.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned7.begin(), muCleaned7.end(), std::back_inserter(muCleaned));
      std::copy(muCleaned8.begin(), muCleaned8.end(), std::back_inserter(muCleaned));

      std::vector<l1t::TrackerMuon> trackMatchedMuonsNoIso = track_mu_match_->convert(muCleaned, 32);

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
      std::vector<l1t::TrackerMuon> sortedTrackMuonsNoIso = track_mu_match_->sort(trackMatchedMuonsNoIso, 12);

      isolation_->isolation_allmu_alltrk(sortedTrackMuonsNoIso, convertedTracks);

      //tauto3mu_->GetTau3Mu(sortedTrackMuonsNoIso, convertedTracks);

      track_mu_match_->outputGT(sortedTrackMuonsNoIso);

      return sortedTrackMuonsNoIso;  //when we add more collections like tau3mu etc we change that
    }

  private:
    int verbose_;
    std::unique_ptr<TrackConverter> tt_track_converter_;
    std::unique_ptr<ROITempAssociator> roi_assoc_;
    std::unique_ptr<TrackMuonMatchAlgorithm> track_mu_match_;
    std::unique_ptr<Isolation> isolation_;
    std::unique_ptr<Tauto3Mu> tauto3mu_;

    std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > associateTracksWithNonant(
        const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks, uint processor) {
      std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > out;
      for (const auto& track : tracks) {
        if (track->phiSector() == processor) {
          out.push_back(track);
        }
      }
      return out;
    }

    l1t::MuonStubRefVector associateStubsWithNonant(const l1t::MuonStubRefVector& allStubs, uint processor) {
      l1t::MuonStubRefVector out;

      ap_int<BITSSTUBCOORD> center = ap_int<BITSSTUBCOORD>((processor * 910) / 32);

      for (const auto& s : allStubs) {
        ap_int<BITSSTUBCOORD> phi = 0;
        if (s->quality() & 0x1)
          phi = s->coord1();
        else
          phi = s->coord2();

        ap_int<BITSSTUBCOORD> deltaPhi = phi - center;
        ap_uint<BITSSTUBCOORD - 1> absDeltaPhi =
            (deltaPhi < 0) ? ap_uint<BITSSTUBCOORD - 1>(-deltaPhi) : ap_uint<BITSSTUBCOORD - 1>(deltaPhi);
        if (absDeltaPhi < 42)
          out.push_back(s);

        /* if (processor==0 && phi>=-3000/32 && phi<=3000/32 ) */
        /*   out.push_back(s); */
        /* else if (processor==1 && (phi>=-1000/32 && phi<=5000/32) ) */
        /*   out.push_back(s); */
        /* else if (processor==2 && (phi>=500/32 && phi<=6500/32) ) */
        /*   out.push_back(s); */
        /* else if (processor==3 && (phi>=2000/32 || phi<=-8000/32) ) */
        /*   out.push_back(s); */
        /* else if (processor==4 && (phi>=4500/32 || phi<=-6000/32) ) */
        /*   out.push_back(s); */
        /* else if (processor==5 && (phi>=6000/32 || phi<=-4500/32) ) */
        /*   out.push_back(s); */
        /* else if (processor==6 && (phi>=8000/32 || phi<=-2000/32) ) */
        /*   out.push_back(s); */
        /* else if (processor==7 && (phi>=-7000/32 && phi<=0) ) */
        /*   out.push_back(s); */
        /* else if (processor==8 && (phi>=-4500/32 && phi<=1000/32) ) */
        /*   out.push_back(s); */
      }
      return out;
    }
  };
}  // namespace Phase2L1GMT

#endif
