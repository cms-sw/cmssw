#include "L1Trigger/L1TMuonEndCap/interface/PtAssignment.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"


void PtAssignment::configure(
    const PtAssignmentEngine* pt_assign_engine,
    int verbose, int endcap, int sector, int bx,
    int ptLUTVersion, bool readPtLUTFile, bool fixMode15HighPt,
    bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt,
    bool bugGMTPhi
) {
  assert(pt_assign_engine != nullptr);

  pt_assign_engine_ = const_cast<PtAssignmentEngine*>(pt_assign_engine);

  verbose_ = verbose;
  endcap_  = endcap;
  sector_  = sector;
  bx_      = bx;

  pt_assign_engine_->configure(
      verbose_,
      ptLUTVersion, readPtLUTFile, fixMode15HighPt,
      bug9BitDPhi, bugMode7CLCT, bugNegPt
  );

  bugGMTPhi_ = bugGMTPhi;
}

void PtAssignment::process(
    EMTFTrackCollection& best_tracks
) {
  using address_t = PtAssignmentEngine::address_t;

  EMTFTrackCollection::iterator best_tracks_it  = best_tracks.begin();
  EMTFTrackCollection::iterator best_tracks_end = best_tracks.end();

  for (; best_tracks_it != best_tracks_end; ++best_tracks_it) {
    EMTFTrack& track = *best_tracks_it;  // pass by reference

    address_t address = 0;
    float     xmlpt   = 0.;
    float     pt      = 0.;
    if (track.NumHits() == 1) {
      address = 0;
      xmlpt   = 0;
      pt      = 5.5 / std::cosh( std::abs(track.Eta()) );
    } else {
      address = pt_assign_engine_->calculate_address(track);
      xmlpt   = pt_assign_engine_->calculate_pt(address);

      // // Un-comment to check address packing / unpacking
      // assert( fabs(xmlpt - pt_assign_engine_->calculate_pt(track)) < 0.001 );

      pt  = (xmlpt < 0.) ? 1. : xmlpt;  // Matt used fabs(-1) when mode is invalid
      pt *= pt_assign_engine_->scale_pt(pt, track.Mode());  // Multiply by some factor to achieve 90% efficiency at threshold
    }

    int gmt_pt = aux().getGMTPt(pt);            // Encode integer pT in GMT format
    pt = (gmt_pt <= 0) ?  0 : (gmt_pt-1) * 0.5; // Decode integer pT (result is in 0.5 GeV step)

    int gmt_phi = aux().getGMTPhi(track.Phi_fp());

    if (!bugGMTPhi_) {
      gmt_phi = aux().getGMTPhiV2(track.Phi_fp());
    }

    int gmt_eta = aux().getGMTEta(track.Theta_fp(), track.Endcap());  // Convert to integer eta using FW LUT

    // Notes from Alex (2016-09-28):
    //
    //     When using two's complement, you get two eta bins with zero coordinate.
    // This peculiarity is created because positive and negative endcaps are
    // processed by separate processors, so each of them gets its own zero bin.
    // With simple inversion, the eta scale becomes uniform, one bin for one
    // eta value.
    bool use_ones_complem_gmt_eta = true;
    if (use_ones_complem_gmt_eta) {
      gmt_eta = (gmt_eta < 0) ? ~(-gmt_eta) : gmt_eta;
    }

    int gmt_quality = 0;
    if (track.NumHits() == 1)
      gmt_quality = log2( track.Mode() );  // 8 --> 3, 4 --> 2, 2 --> 1, 1 --> 0
    else
      gmt_quality = aux().getGMTQuality(track.Mode(), track.Theta_fp());

    std::pair<int, int> gmt_charge = std::make_pair(0, 0);
    if (track.NumHits() == 1) {
      int CLCT = track.Hits().at(0).Pattern();
      if (CLCT != 10) {
        if (endcap_ == 1)
          gmt_charge = std::make_pair( (CLCT % 2) == 0 ? 0 : 1, 1);
        else
          gmt_charge = std::make_pair( (CLCT % 2) == 0 ? 1 : 0, 1);
      }
    } else {
      std::vector<int> phidiffs;
      for (int i = 0; i < NUM_STATION_PAIRS; ++i) {
        int phidiff = (track.PtLUT().sign_ph[i] == 1) ? track.PtLUT().delta_ph[i] : -track.PtLUT().delta_ph[i];
        phidiffs.push_back(phidiff);
      }
      gmt_charge = aux().getGMTCharge(track.Mode(), phidiffs);
    }

    // _________________________________________________________________________
    // Output

    EMTFPtLUT tmp_LUT = track.PtLUT();
    tmp_LUT.address   = address;

    track.set_PtLUT  ( tmp_LUT );
    track.set_pt_XML ( xmlpt );
    track.set_pt     ( pt );
    track.set_charge ( (gmt_charge.second == 1) ? ((gmt_charge.first == 1) ? -1 : +1) : 0 );

    track.set_gmt_pt           ( gmt_pt );
    track.set_gmt_phi          ( gmt_phi );
    track.set_gmt_eta          ( gmt_eta );
    track.set_gmt_quality      ( gmt_quality );
    track.set_gmt_charge       ( gmt_charge.first );
    track.set_gmt_charge_valid ( gmt_charge.second );
  }

  if (verbose_ > 0) {  // debug
    for (const auto& track: best_tracks) {
      std::cout << "track: " << track.Winner() << " pt address: " << track.PtLUT().address
          << " GMT pt: " << track.GMT_pt() << " pt: " << track.Pt() << " mode: " << track.Mode()
          << " GMT charge: " << track.GMT_charge() << " quality: " << track.GMT_quality()
          << " eta: " << track.GMT_eta() << " phi: " << track.GMT_phi()
          << std::endl;
    }
  }
}

const PtAssignmentEngineAux& PtAssignment::aux() const {
  return pt_assign_engine_->aux();
}
