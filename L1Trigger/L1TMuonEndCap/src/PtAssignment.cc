#include "L1Trigger/L1TMuonEndCap/interface/PtAssignment.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineDxy.h"

void PtAssignment::configure(PtAssignmentEngine* pt_assign_engine,
                             PtAssignmentEngineDxy* pt_assign_engine_dxy,
                             int verbose,
                             int endcap,
                             int sector,
                             int bx,
                             bool readPtLUTFile,
                             bool fixMode15HighPt,
                             bool bug9BitDPhi,
                             bool bugMode7CLCT,
                             bool bugNegPt,
                             bool bugGMTPhi,
                             bool promoteMode7,
                             int modeQualVer,
                             std::string pbFileName) {
  emtf_assert(pt_assign_engine != nullptr);
  emtf_assert(pt_assign_engine_dxy != nullptr);

  pt_assign_engine_ = pt_assign_engine;

  pt_assign_engine_dxy_ = pt_assign_engine_dxy;

  verbose_ = verbose;
  endcap_ = endcap;
  sector_ = sector;
  bx_ = bx;

  pt_assign_engine_->configure(verbose_, readPtLUTFile, fixMode15HighPt, bug9BitDPhi, bugMode7CLCT, bugNegPt);

  pt_assign_engine_dxy_->configure(verbose_, pbFileName);

  bugGMTPhi_ = bugGMTPhi;
  promoteMode7_ = promoteMode7;
  modeQualVer_ = modeQualVer;
}

void PtAssignment::process(EMTFTrackCollection& best_tracks) {
  using address_t = PtAssignmentEngine::address_t;

  EMTFTrackCollection::iterator best_tracks_it = best_tracks.begin();
  EMTFTrackCollection::iterator best_tracks_end = best_tracks.end();

  for (; best_tracks_it != best_tracks_end; ++best_tracks_it) {
    EMTFTrack& track = *best_tracks_it;  // pass by reference

    // Assign GMT eta and phi
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

    // Assign prompt & displaced pT
    address_t address = 0;
    float xmlpt = 0.;
    float pt = 0.;
    int gmt_pt = 0;

    float pt_dxy = 0.;
    float dxy = 0.;
    int gmt_pt_dxy = 0;
    int gmt_dxy = 0;

    if (track.Mode() != 1) {
      address = pt_assign_engine_->calculate_address(track);
      xmlpt = pt_assign_engine_->calculate_pt(address);

      // Check address packing / unpacking using PtAssignmentEngine2017::calculate_pt_xml(const EMTFTrack& track)
      if (pt_assign_engine_->get_pt_lut_version() > 5 &&
          not(fabs(xmlpt - pt_assign_engine_->calculate_pt(track)) < 0.001)) {
        edm::LogError("L1T") << "EMTF pT assignment mismatch: xmlpt = " << xmlpt
                             << ", pt_assign_engine_->calculate_pt(track)) = "
                             << pt_assign_engine_->calculate_pt(track);
      }

      pt = (xmlpt < 0.) ? 1. : xmlpt;  // Matt used fabs(-1) when mode is invalid
      pt *= pt_assign_engine_->scale_pt(
          pt, track.Mode());  // Multiply by some factor to achieve 90% efficiency at threshold

      gmt_pt = aux().getGMTPt(pt);  // Encode integer pT in GMT format
    }                               // End if (track.Mode() != 1)
    else {
      gmt_pt = 10 - (abs(gmt_eta) / 32);
    }

    pt = (gmt_pt <= 0) ? 0 : (gmt_pt - 1) * 0.5;  // Decode integer pT (result is in 0.5 GeV step)

    // Calculate displaced pT and d0 using NN
    emtf::Feature feature;
    emtf::Prediction prediction;

    feature.fill(0);
    prediction.fill(0);

    pt_assign_engine_dxy_->calculate_pt_dxy(track, feature, prediction);

    pt_dxy = std::abs(prediction.at(0));
    dxy = prediction.at(1);

    gmt_pt_dxy = aux().getGMTPtDxy(pt_dxy);
    gmt_dxy = aux().getGMTDxy(dxy);

    pt_dxy = aux().getPtFromGMTPtDxy(gmt_pt_dxy);

    int gmt_quality = 0;
    if (track.Mode() != 1) {
      gmt_quality = aux().getGMTQuality(track.Mode(), track.Theta_fp(), promoteMode7_, modeQualVer_);
    } else {  // Special quality for single-hit tracks from ME1/1
      gmt_quality = track.Hits().front().Pattern() / 4;
    }

    std::pair<int, int> gmt_charge = std::make_pair(0, 0);
    if (track.Mode() != 1) {
      std::vector<int> phidiffs;
      for (int i = 0; i < emtf::NUM_STATION_PAIRS; ++i) {
        int phidiff = (track.PtLUT().sign_ph[i] == 1) ? track.PtLUT().delta_ph[i] : -track.PtLUT().delta_ph[i];
        phidiffs.push_back(phidiff);
      }
      gmt_charge = aux().getGMTCharge(track.Mode(), phidiffs);
    } else {  // Special charge assignment for single-hit tracks from ME1/1
      int CLCT = track.Hits().front().Pattern();
      if (CLCT != 10) {
        if (endcap_ == 1)
          gmt_charge = std::make_pair((CLCT % 2) == 0 ? 0 : 1, 1);
        else
          gmt_charge = std::make_pair((CLCT % 2) == 0 ? 1 : 0, 1);
      }
    }

    // _________________________________________________________________________
    // Output

    EMTFPtLUT tmp_LUT = track.PtLUT();
    tmp_LUT.address = address;

    track.set_PtLUT(tmp_LUT);
    track.set_pt_XML(xmlpt);
    track.set_pt(pt);
    track.set_pt_dxy(pt_dxy);
    track.set_dxy(dxy);
    track.set_charge((gmt_charge.second == 1) ? ((gmt_charge.first == 1) ? -1 : +1) : 0);

    track.set_gmt_pt(gmt_pt);
    track.set_gmt_pt_dxy(gmt_pt_dxy);
    track.set_gmt_dxy(gmt_dxy);
    track.set_gmt_phi(gmt_phi);
    track.set_gmt_eta(gmt_eta);
    track.set_gmt_quality(gmt_quality);
    track.set_gmt_charge(gmt_charge.first);
    track.set_gmt_charge_valid(gmt_charge.second);
  }

  // Remove worst track if it addresses the same bank as one of two best tracks
  bool disable_worst_track_in_same_bank = true;
  if (disable_worst_track_in_same_bank) {
    // FW macro for detecting same bank address
    // bank and chip must match, and valid flags must be set
    // a and b are indexes 0,1,2
    // `define sb(a,b) (ptlut_addr[a][29:26] == ptlut_addr[b][29:26] && ptlut_addr[a][5:2] == ptlut_addr[b][5:2] && ptlut_addr_val[a] && ptlut_addr_val[b])
    auto is_in_same_bank = [](const EMTFTrack& lhs, const EMTFTrack& rhs) {
      unsigned lhs_addr = lhs.PtLUT().address;
      unsigned rhs_addr = rhs.PtLUT().address;
      unsigned lhs_addr_1 = (lhs_addr >> 26) & 0xF;
      unsigned rhs_addr_1 = (rhs_addr >> 26) & 0xF;
      unsigned lhs_addr_2 = (lhs_addr >> 2) & 0xF;
      unsigned rhs_addr_2 = (rhs_addr >> 2) & 0xF;
      return (lhs_addr_1 == rhs_addr_1) && (lhs_addr_2 == rhs_addr_2);
    };

    emtf_assert(best_tracks.size() <= 3);
    if (best_tracks.size() == 3) {
      bool same_bank = is_in_same_bank(best_tracks.at(0), best_tracks.at(2)) ||
                       is_in_same_bank(best_tracks.at(1), best_tracks.at(2));
      if (same_bank) {
        // Set worst track pT to zero
        best_tracks.at(2).set_pt(0);
        best_tracks.at(2).set_gmt_pt(0);
      }
    }
  }

  if (verbose_ > 0) {  // debug
    for (const auto& track : best_tracks) {
      std::cout << "track: " << track.Winner() << " pt address: " << track.PtLUT().address
                << " GMT pt: " << track.GMT_pt() << " pt: " << track.Pt() << " mode: " << track.Mode()
                << " GMT charge: " << track.GMT_charge() << " quality: " << track.GMT_quality()
                << " eta: " << track.GMT_eta() << " phi: " << track.GMT_phi() << std::endl;
    }
  }
}

const PtAssignmentEngineAux& PtAssignment::aux() const { return pt_assign_engine_->aux(); }
