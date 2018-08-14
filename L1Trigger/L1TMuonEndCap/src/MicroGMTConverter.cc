#include "L1Trigger/L1TMuonEndCap/interface/MicroGMTConverter.h"


MicroGMTConverter::MicroGMTConverter() {

}

MicroGMTConverter::~MicroGMTConverter() {

}

void MicroGMTConverter::convert(
    const int global_event_BX,
    const EMTFTrack& in_track,
    l1t::RegionalMuonCand& out_cand
) const {
  l1t::tftype tftype = (in_track.Endcap() == 1) ? l1t::tftype::emtf_pos : l1t::tftype::emtf_neg;
  int sector = in_track.Sector() - 1;

  out_cand.setHwPt            ( in_track.GMT_pt() );
  out_cand.setHwPhi           ( in_track.GMT_phi() );
  out_cand.setHwEta           ( in_track.GMT_eta() );
  out_cand.setHwSign          ( in_track.GMT_charge() );
  out_cand.setHwSignValid     ( in_track.GMT_charge_valid() );
  out_cand.setHwQual          ( in_track.GMT_quality() );
  out_cand.setHwHF            ( false );  // EMTF: halo -> 1
  out_cand.setTFIdentifiers   ( sector, tftype );

  const EMTFPtLUT& ptlut_data = in_track.PtLUT();

  // Form track sub addresses
  int me1_ch_id = (ptlut_data.bt_vi[0] == 0 && ptlut_data.bt_vi[1] != 0) ? ptlut_data.bt_ci[1]+16 : ptlut_data.bt_ci[0];
  int me2_ch_id = ptlut_data.bt_ci[2];
  int me3_ch_id = ptlut_data.bt_ci[3];
  int me4_ch_id = ptlut_data.bt_ci[4];

  int me1_seg_id = (ptlut_data.bt_vi[0] == 0 && ptlut_data.bt_vi[1] != 0) ? ptlut_data.bt_si[1] : ptlut_data.bt_si[0];
  int me2_seg_id = ptlut_data.bt_si[2];
  int me3_seg_id = ptlut_data.bt_si[3];
  int me4_seg_id = ptlut_data.bt_si[4];

  auto get_gmt_chamber_me1 = [](int ch) {
    int gmt_ch = 0;
    if (ch == 10)
      gmt_ch = 1;
    else if (ch == 11)
      gmt_ch = 2;
    else if (ch == 12)
      gmt_ch = 3;
    else if (ch == 3+16)
      gmt_ch = 4;
    else if (ch == 6+16)
      gmt_ch = 5;
    else if (ch == 9+16)
      gmt_ch = 6;
    return gmt_ch;
  };

  auto get_gmt_chamber = [](int ch) {
    int gmt_ch = 0;
    if (ch == 10)
      gmt_ch = 1;
    else if (ch == 11)
      gmt_ch = 2;
    else if (ch == 3)
      gmt_ch = 3;
    else if (ch == 9)
      gmt_ch = 4;
    return gmt_ch;
  };

  // Truncate kBX to 11 bits and offset by 25 from global event BX in EMTF firmware
  int EMTF_kBX = (abs(global_event_BX) % 2048) - 25 + in_track.BX();
  if (EMTF_kBX < 0) EMTF_kBX += 2048;

  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME1Seg, me1_seg_id);
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME1Ch , get_gmt_chamber_me1(me1_ch_id));
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME2Seg, me2_seg_id);
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME2Ch , get_gmt_chamber(me2_ch_id));
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME3Seg, me3_seg_id);
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME3Ch , get_gmt_chamber(me3_ch_id));
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME4Seg, me4_seg_id);
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kME4Ch , get_gmt_chamber(me4_ch_id));
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kTrkNum, in_track.Track_num());
  out_cand.setTrackSubAddress(l1t::RegionalMuonCand::kBX    , EMTF_kBX);
}

void MicroGMTConverter::convert_all(
    const edm::Event& iEvent,
    const EMTFTrackCollection& in_tracks,
    l1t::RegionalMuonCandBxCollection& out_cands
) const {
  int gmtMinBX = -2;
  int gmtMaxBX = +2;

  out_cands.clear();
  out_cands.setBXRange(gmtMinBX, gmtMaxBX);

  for (const auto& in_track : in_tracks) {
    int bx = in_track.BX();

    if (gmtMinBX <= bx && bx <= gmtMaxBX) {
      l1t::RegionalMuonCand out_cand;

      convert(iEvent.bunchCrossing(), in_track, out_cand);
      out_cands.push_back(bx, out_cand);
    }
  }

  // Sort by processor to match uGMT unpacked order
  emtf::sort_uGMT_muons(out_cands);

}


namespace emtf {

void sort_uGMT_muons(
   l1t::RegionalMuonCandBxCollection& cands
) {

  int minBX = cands.getFirstBX();
  int maxBX = cands.getLastBX();
  int emtfMinProc =  0; // ME+ sector 1
  int emtfMaxProc = 11; // ME- sector 6

  // New collection, sorted by processor to match uGMT unpacked order
  auto sortedCands = std::make_unique<l1t::RegionalMuonCandBxCollection>();
  sortedCands->clear();
  sortedCands->setBXRange(minBX, maxBX);
  for (int iBX = minBX; iBX <= maxBX; ++iBX) {
    for (int proc = emtfMinProc; proc <= emtfMaxProc; proc++) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator cand = cands.begin(iBX); cand != cands.end(iBX); ++cand) {
        int cand_proc = cand->processor();
        if (cand->trackFinderType() == l1t::tftype::emtf_neg) cand_proc += 6;
        if (cand_proc != proc) continue;
        sortedCands->push_back(iBX, *cand);
      }
    }
  }

  // Return sorted collection
  std::swap(cands, *sortedCands);
  sortedCands.reset();
}

} // End namespace emtf
