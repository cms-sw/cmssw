#ifndef PHASE2GMT_TRACKMUONMATCHALGO
#define PHASE2GMT_TRACKMUONMATCHALGO

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"
#include "TrackConverter.h"
#include "PreTrackMatchedMuon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MuonROI.h"
#include "ConvertedTTTrack.h"
#include "Constants.h"
#include <fstream>

namespace Phase2L1GMT {

  const unsigned int PHIDIVIDER = 1 << (BITSPHI - BITSSTUBCOORD);
  const unsigned int ETADIVIDER = 1 << (BITSETA - BITSSTUBETA);

  typedef struct {
    ap_int<BITSSTUBCOORD> coord1;
    ap_uint<BITSSIGMACOORD> sigma_coord1;
    ap_int<BITSSTUBCOORD> coord2;
    ap_uint<BITSSIGMACOORD> sigma_coord2;
    ap_int<BITSSTUBETA> eta;
    ap_uint<BITSSIGMAETA> sigma_eta1;
    ap_uint<BITSSIGMAETA> sigma_eta2;
    ap_uint<1> valid;
    ap_uint<1> is_barrel;
  } propagation_t;

  typedef struct {
    ap_uint<BITSMATCHQUALITY - 2> quality;
    ap_uint<BITSSTUBID> id;
    ap_uint<1> valid;
    bool isGlobal;
    l1t::RegionalMuonCandRef muRef;
    l1t::MuonStubRef stubRef;

  } match_t;

  class TrackMuonMatchAlgorithm {
  public:
    TrackMuonMatchAlgorithm(const edm::ParameterSet& iConfig) : verbose_(iConfig.getParameter<int>("verbose")) {}

    ~TrackMuonMatchAlgorithm() {}

    std::vector<PreTrackMatchedMuon> processNonant(const std::vector<ConvertedTTTrack>& convertedTracks,
                                                   const std::vector<MuonROI>& rois) {
      std::vector<PreTrackMatchedMuon> preMuons;
      for (const auto& track : convertedTracks) {
        PreTrackMatchedMuon mu = processTrack(track, rois);
        if (mu.valid() && preMuons.size() < 16)
          preMuons.push_back(mu);
      }
      std::vector<PreTrackMatchedMuon> cleanedMuons = clean(preMuons);
      return cleanedMuons;
    }

    std::vector<PreTrackMatchedMuon> cleanNeighbor(const std::vector<PreTrackMatchedMuon>& muons,
                                                   const std::vector<PreTrackMatchedMuon>& muonsPrevious,
                                                   const std::vector<PreTrackMatchedMuon>& muonsNext,
                                                   bool equality) {
      std::vector<PreTrackMatchedMuon> out;

      if (muons.empty())
        return out;

      if (verbose_ == 1) {
        printf("-----Cleaning Up Muons in the neighbours\n");
        printf("Before:\n");
      }

      for (uint i = 0; i < muons.size(); ++i) {
        if (verbose_ == 1) {
          muons[i].print();
        }
        ap_uint<5> mask = 0x1f;
        for (uint j = 0; j < muonsPrevious.size(); ++j) {
          mask = mask & cleanMuon(muons[i], muonsPrevious[j], equality);
        }
        for (uint j = 0; j < muonsNext.size(); ++j) {
          mask = mask & cleanMuon(muons[i], muonsNext[j], equality);
        }
        if (mask) {
          if (verbose_ == 1)
            printf("kept\n");
          out.push_back(muons[i]);
        } else {
          if (verbose_ == 1)
            printf("discarded\n");
        }
      }
      return out;
    }

    std::vector<l1t::TrackerMuon> convert(std::vector<PreTrackMatchedMuon>& muons, uint maximum) {
      std::vector<l1t::TrackerMuon> out;
      for (const auto& mu : muons) {
        if (out.size() == maximum)
          break;
        l1t::TrackerMuon muon(mu.trkPtr(), mu.charge(), mu.pt(), mu.eta(), mu.phi(), mu.z0(), mu.d0(), mu.quality());
        muon.setMuonRef(mu.muonRef());
        for (const auto& stub : mu.stubs())
          muon.addStub(stub);
        out.push_back(muon);
        if (verbose_ == 1) {
          printf("Final Muon:");
          muon.print();
        }
      }
      return out;
    }

    bool outputGT(std::vector<l1t::TrackerMuon>& muons) {
      for (auto& mu : muons) {
        wordtype word1 = 0;
        wordtype word2 = 0;

        int bstart = 0;
        bstart = wordconcat<wordtype>(word1, bstart, mu.hwPt(), BITSGTPT);
        bstart = wordconcat<wordtype>(word1, bstart, mu.hwPhi(), BITSGTPHI);
        bstart = wordconcat<wordtype>(word1, bstart, mu.hwEta(), BITSGTETA);
        bstart = wordconcat<wordtype>(word1, bstart, mu.hwZ0(), BITSGTZ0);
        bstart = wordconcat<wordtype>(word1, bstart, (mu.hwD0() >> 2), BITSGTD0);

        bstart = 0;
        bstart = wordconcat<wordtype>(word2, bstart, mu.hwCharge(), 1);
        bstart = wordconcat<wordtype>(word2, bstart, mu.hwQual(), BITSGTQUAL);
        bstart = wordconcat<wordtype>(word2, bstart, mu.hwIso(), BITSGTISO);
        bstart = wordconcat<wordtype>(word2, bstart, mu.hwBeta(), BITSGTBETA);

        std::array<uint64_t, 2> wordout = {{word1, word2}};
        mu.setWord(wordout);
      }
      return true;
    }

    std::vector<l1t::TrackerMuon> sort(std::vector<l1t::TrackerMuon>& muons, uint maximum) {
      if (muons.size() < 2)
        return muons;

      std::sort(muons.begin(), muons.end(), [](l1t::TrackerMuon a, l1t::TrackerMuon b) { return a.hwPt() > b.hwPt(); });
      std::vector<l1t::TrackerMuon> out;
      for (unsigned int i = 0; i < muons.size(); ++i) {
        out.push_back(muons[i]);
        if (i == (maximum - 1))
          break;
      }

      return out;
    }

  private:
    int verbose_;

    propagation_t propagate(const ConvertedTTTrack& track, uint layer) {
      ap_uint<BITSPROPCOORD> prop_coord1 = 0;
      ap_uint<BITSPROPCOORD> prop_coord2 = 0;
      ap_uint<BITSPROPSIGMACOORD_A> res0_coord1 = 0;
      ap_uint<BITSPROPSIGMACOORD_B> res1_coord1 = 0;
      ap_uint<BITSPROPSIGMACOORD_A> res0_coord2 = 0;
      ap_uint<BITSPROPSIGMACOORD_B> res1_coord2 = 0;
      ap_uint<BITSPROPSIGMAETA_A> res0_eta1 = 0;
      ap_uint<BITSPROPSIGMAETA_B> res1_eta = 0;
      ap_uint<BITSPROPSIGMAETA_A> res0_eta2 = 0;
      ap_uint<1> is_barrel = 0;

      uint reducedAbsEta = track.abseta() / 8;

      if (layer == 0) {
        prop_coord1 = lt_prop_coord1_0[reducedAbsEta];
        prop_coord2 = lt_prop_coord2_0[reducedAbsEta];
        res0_coord1 = lt_res0_coord1_0[reducedAbsEta];
        res1_coord1 = lt_res1_coord1_0[reducedAbsEta];
        res0_coord2 = lt_res0_coord2_0[reducedAbsEta];
        res1_coord2 = lt_res1_coord2_0[reducedAbsEta];
        res0_eta1 = lt_res0_eta1_0[reducedAbsEta];
        res1_eta = lt_res1_eta_0[reducedAbsEta];
        res0_eta2 = lt_res0_eta2_0[reducedAbsEta];
        is_barrel = reducedAbsEta < barrelLimit0_ ? 1 : 0;
      } else if (layer == 1) {
        prop_coord1 = lt_prop_coord1_1[reducedAbsEta];
        prop_coord2 = lt_prop_coord2_1[reducedAbsEta];
        res0_coord1 = lt_res0_coord1_1[reducedAbsEta];
        res1_coord1 = lt_res1_coord1_1[reducedAbsEta];
        res0_coord2 = lt_res0_coord2_1[reducedAbsEta];
        res1_coord2 = lt_res1_coord2_1[reducedAbsEta];
        res0_eta1 = lt_res0_eta1_1[reducedAbsEta];
        res1_eta = lt_res1_eta_1[reducedAbsEta];
        res0_eta2 = lt_res0_eta2_1[reducedAbsEta];
        is_barrel = reducedAbsEta < barrelLimit1_ ? 1 : 0;

      } else if (layer == 2) {
        prop_coord1 = lt_prop_coord1_2[reducedAbsEta];
        prop_coord2 = lt_prop_coord2_2[reducedAbsEta];
        res0_coord1 = lt_res0_coord1_2[reducedAbsEta];
        res1_coord1 = lt_res1_coord1_2[reducedAbsEta];
        res0_coord2 = lt_res0_coord2_2[reducedAbsEta];
        res1_coord2 = lt_res1_coord2_2[reducedAbsEta];
        res0_eta1 = lt_res0_eta1_2[reducedAbsEta];
        res1_eta = lt_res1_eta_2[reducedAbsEta];
        res0_eta2 = lt_res0_eta2_2[reducedAbsEta];
        is_barrel = reducedAbsEta < barrelLimit2_ ? 1 : 0;

      } else if (layer == 3) {
        prop_coord1 = lt_prop_coord1_3[reducedAbsEta];
        prop_coord2 = lt_prop_coord2_3[reducedAbsEta];
        res0_coord1 = lt_res0_coord1_3[reducedAbsEta];
        res1_coord1 = lt_res1_coord1_3[reducedAbsEta];
        res0_coord2 = lt_res0_coord2_3[reducedAbsEta];
        res1_coord2 = lt_res1_coord2_3[reducedAbsEta];
        res0_eta1 = lt_res0_eta1_3[reducedAbsEta];
        res1_eta = lt_res1_eta_3[reducedAbsEta];
        res0_eta2 = lt_res0_eta2_3[reducedAbsEta];
        is_barrel = reducedAbsEta < barrelLimit3_ ? 1 : 0;

      } else if (layer == 4) {
        prop_coord1 = lt_prop_coord1_4[reducedAbsEta];
        prop_coord2 = lt_prop_coord2_4[reducedAbsEta];
        res0_coord1 = lt_res0_coord1_4[reducedAbsEta];
        res1_coord1 = lt_res1_coord1_4[reducedAbsEta];
        res0_coord2 = lt_res0_coord2_4[reducedAbsEta];
        res1_coord2 = lt_res1_coord2_4[reducedAbsEta];
        res0_eta1 = lt_res0_eta1_4[reducedAbsEta];
        res1_eta = lt_res1_eta_4[reducedAbsEta];
        res0_eta2 = lt_res0_eta2_4[reducedAbsEta];
        is_barrel = 0;
      }

      propagation_t out;
      ap_int<BITSTTCURV> curvature = track.curvature();
      ap_int<BITSPHI> phi = track.phi();
      ap_int<BITSPROPCOORD + BITSTTCURV> c1kFull = prop_coord1 * curvature;
      ap_int<BITSPROPCOORD + BITSTTCURV - 10> c1k = (c1kFull) / 1024;
      ap_int<BITSPHI> coord1 = phi - c1k;

      out.coord1 = coord1 / PHIDIVIDER;

      ap_int<BITSPROPCOORD + BITSTTCURV> c2kFull = prop_coord2 * curvature;

      ap_int<BITSPROPCOORD + BITSTTCURV - 10> c2k = (c2kFull) / 1024;
      if (is_barrel)
        out.coord2 = -c2k / PHIDIVIDER;
      else
        out.coord2 = (phi - c2k) / PHIDIVIDER;

      ap_int<BITSETA> eta = track.eta();
      out.eta = eta / ETADIVIDER;

      ap_uint<2 * BITSTTCURV - 2> curvature2All = curvature * curvature;
      ap_uint<BITSTTCURV2> curvature2 = curvature2All / 2;

      //Remember to change emulator with new k2
      ap_uint<BITSPROPSIGMACOORD_B + BITSTTCURV2> rescoord1k = (res1_coord1 * curvature2) >> 23;
      ap_ufixed<BITSSIGMACOORD, BITSSIGMACOORD, AP_TRN_ZERO, AP_SAT_SYM> sigma_coord1 = res0_coord1 + rescoord1k;
      out.sigma_coord1 = ap_uint<BITSSIGMACOORD>(sigma_coord1);

      ap_uint<BITSPROPSIGMACOORD_B + BITSTTCURV2> rescoord2k = (res1_coord2 * curvature2) >> 23;
      ap_ufixed<BITSSIGMACOORD, BITSSIGMACOORD, AP_TRN_ZERO, AP_SAT_SYM> sigma_coord2 = res0_coord2 + rescoord2k;
      out.sigma_coord2 = ap_uint<BITSSIGMACOORD>(sigma_coord2);

      ap_uint<BITSPROPSIGMAETA_B + BITSTTCURV2> resetak = (res1_eta * curvature2) >> 23;
      ap_ufixed<BITSSIGMAETA, BITSSIGMAETA, AP_TRN_ZERO, AP_SAT_SYM> sigma_eta1 = res0_eta1 + resetak;
      out.sigma_eta1 = ap_uint<BITSSIGMAETA>(sigma_eta1);
      ap_ufixed<BITSSIGMAETA, BITSSIGMAETA, AP_TRN_ZERO, AP_SAT_SYM> sigma_eta2 = res0_eta2 + resetak;
      out.sigma_eta2 = ap_uint<BITSSIGMAETA>(sigma_eta2);
      out.valid = 1;
      out.is_barrel = is_barrel;

      if (verbose_ == 1)

        printf("Propagating to layer %d:is barrel=%d  coords=%d+-%d , %d +-%d etas = %d +- %d +-%d\n",
               int(layer),
               out.is_barrel.to_int(),
               out.coord1.to_int(),
               out.sigma_coord1.to_int(),
               out.coord2.to_int(),
               out.sigma_coord2.to_int(),
               out.eta.to_int(),
               out.sigma_eta1.to_int(),
               out.sigma_eta2.to_int());

      return out;
    }

    ap_uint<BITSSIGMAETA + 1> deltaEta(const ap_int<BITSSTUBETA>& eta1, const ap_int<BITSSTUBETA>& eta2) {
      ap_fixed<BITSSIGMAETA + 2, BITSSIGMAETA + 2, AP_TRN_ZERO, AP_SAT_SYM> dEta = eta1 - eta2;
      if (dEta < 0)
        return ap_uint<BITSSIGMAETA + 1>(-dEta);
      else
        return ap_uint<BITSSIGMAETA + 1>(dEta);
    }

    ap_uint<BITSSIGMACOORD + 1> deltaCoord(const ap_int<BITSSTUBCOORD>& phi1, const ap_int<BITSSTUBCOORD>& phi2) {
      ap_int<BITSSTUBCOORD> dPhiRoll = phi1 - phi2;
      ap_ufixed<BITSSIGMACOORD + 1, BITSSIGMACOORD + 1, AP_TRN_ZERO, AP_SAT_SYM> dPhi;
      if (dPhiRoll < 0)
        dPhi = ap_ufixed<BITSSIGMACOORD + 1, BITSSIGMACOORD + 1, AP_TRN_ZERO, AP_SAT_SYM>(-dPhiRoll);
      else
        dPhi = ap_ufixed<BITSSIGMACOORD + 1, BITSSIGMACOORD + 1, AP_TRN_ZERO, AP_SAT_SYM>(dPhiRoll);

      return ap_uint<BITSSIGMACOORD + 1>(dPhi);
    }

    match_t match(const propagation_t prop, const l1t::MuonStubRef& stub) {
      if (verbose_ == 1) {
        printf("Matching to ");
        stub->print();
      }
      //Matching of Coord1
      ap_uint<1> coord1Matched;
      ap_uint<BITSSIGMACOORD + 1> deltaCoord1 = deltaCoord(prop.coord1, stub->coord1());
      if (deltaCoord1 <= prop.sigma_coord1 && (stub->quality() & 0x1)) {
        coord1Matched = 1;
      } else {
        coord1Matched = 0;
      }
      if (verbose_ == 1)
        printf("Coord1 matched=%d delta=%d res=%d\n",
               coord1Matched.to_int(),
               deltaCoord1.to_int(),
               prop.sigma_coord1.to_int());

      //Matching of Coord2
      ap_uint<1> coord2Matched;
      ap_uint<BITSSIGMACOORD + 1> deltaCoord2 = deltaCoord(prop.coord2, stub->coord2());
      if (deltaCoord2 <= prop.sigma_coord2 && (stub->quality() & 0x2)) {
        coord2Matched = 1;
      } else {
        coord2Matched = 0;
      }
      if (verbose_ == 1)
        printf("Coord2 matched=%d delta=%d res=%d\n",
               coord2Matched.to_int(),
               deltaCoord2.to_int(),
               prop.sigma_coord2.to_int());

      //Matching of Eta1

      ap_uint<1> eta1Matched;

      //if we have really bad quality[Barrel no eta]
      //increase the resolution
      ap_ufixed<BITSSIGMAETA, BITSSIGMAETA, AP_TRN_ZERO, AP_SAT_SYM> prop_sigma_eta1;
      if (stub->etaQuality() == 0)
        prop_sigma_eta1 = prop.sigma_eta1 + 6;
      else
        prop_sigma_eta1 = prop.sigma_eta1;

      ap_uint<BITSSIGMAETA + 1> deltaEta1 = deltaEta(prop.eta, stub->eta1());
      if (deltaEta1 <= prop_sigma_eta1 && (stub->etaQuality() == 0 || (stub->etaQuality() & 0x1)))
        eta1Matched = 1;
      else
        eta1Matched = 0;

      if (verbose_ == 1)
        printf("eta1 matched=%d delta=%d res=%d\n", eta1Matched.to_int(), deltaEta1.to_int(), prop_sigma_eta1.to_int());

      //Matching of Eta2

      ap_uint<1> eta2Matched;

      ap_uint<BITSSIGMAETA + 1> deltaEta2 = deltaEta(prop.eta, stub->eta2());
      if (deltaEta2 <= prop.sigma_eta2 && (stub->etaQuality() & 0x2))
        eta2Matched = 1;
      else
        eta2Matched = 0;
      match_t out;
      out.id = stub->id();

      if (verbose_ == 1)
        printf("eta2 matched=%d delta=%d res=%d\n", eta2Matched.to_int(), deltaEta2.to_int(), prop.sigma_eta2.to_int());

      //if barrel, coord1 has to always be matched, coord2 maybe and eta1 is needed if etaQ=0 or then the one that depends on eta quality
      if (prop.is_barrel) {
        out.valid = (coord1Matched == 1 && (eta1Matched == 1 || eta2Matched == 1));
        if (out.valid == 0) {
          out.quality = 0;
        } else {
          out.quality = 32 - deltaCoord1;
          if (coord2Matched == 1)
            out.quality += 32 - deltaCoord2;
        }
      }
      //if endcap each coordinate is independent except the case where phiQuality=1 and etaQuality==3
      else {
        bool match1 = (coord1Matched == 1 && eta1Matched == 1);
        bool match2 = (coord2Matched == 1 && eta2Matched == 1);
        bool match3 =
            (coord1Matched == 1 && (eta1Matched || eta2Matched) && stub->etaQuality() == 3 && stub->quality() == 1);
        out.valid = match1 || match2 || match3;
        if (out.valid == 0)
          out.quality = 0;
        else {
          out.quality = 0;
          if (match1 || match3)
            out.quality += 32 - deltaCoord1;
          if (match2)
            out.quality += 32 - deltaCoord2;
        }
      }
      if (verbose_ == 1)
        printf("GlobalMatchQuality = %d\n", out.quality.to_int());
      out.stubRef = stub;
      return out;
    }

    match_t propagateAndMatch(const ConvertedTTTrack& track, const l1t::MuonStubRef& stub) {
      propagation_t prop = propagate(track, stub->tfLayer());
      return match(prop, stub);
    }

    match_t getBest(const std::vector<match_t> matches) {
      match_t best = matches[0];
      for (const auto& m : matches) {
        if (m.quality > best.quality)
          best = m;
      }

      return best;
    }

    PreTrackMatchedMuon processTrack(const ConvertedTTTrack& track, const std::vector<MuonROI>& rois) {
      std::vector<match_t> matchInfo0;
      std::vector<match_t> matchInfo1;
      std::vector<match_t> matchInfo2;
      std::vector<match_t> matchInfo3;
      std::vector<match_t> matchInfo4;

      if (verbose_ == 1 && !rois.empty()) {
        printf("-----------processing new track----------");
        track.print();
      }
      for (const auto& roi : rois) {
        if (verbose_ == 1) {
          printf("New ROI with %d stubs \n", int(roi.stubs().size()));
        }
        for (const auto& stub : roi.stubs()) {
          match_t m = propagateAndMatch(track, stub);
          if (m.valid == 1) {
            if (roi.isGlobalMuon() && roi.muonRef().isNonnull()) {
              m.isGlobal = true;
              m.muRef = roi.muonRef();
            }

            if (stub->tfLayer() == 0)
              matchInfo0.push_back(m);
            else if (stub->tfLayer() == 1)
              matchInfo1.push_back(m);
            else if (stub->tfLayer() == 2)
              matchInfo2.push_back(m);
            else if (stub->tfLayer() == 3)
              matchInfo3.push_back(m);
            else if (stub->tfLayer() == 4)
              matchInfo4.push_back(m);
          }
        }
      }

      ap_ufixed<6, 6, AP_TRN_ZERO, AP_SAT_SYM> ptPenalty = ap_ufixed<6, 6, AP_TRN_ZERO, AP_SAT_SYM>(track.pt() / 32);

      ap_uint<BITSMATCHQUALITY> quality = 0;
      PreTrackMatchedMuon muon(track.charge(), track.pt(), track.eta(), track.phi(), track.z0(), track.d0());

      if (!matchInfo0.empty()) {
        match_t b = getBest(matchInfo0);
        if (b.valid) {
          muon.addStub(b.stubRef);
          if (b.isGlobal)
            muon.addMuonRef(b.muRef);
          quality += b.quality;
        }
      }
      if (!matchInfo1.empty()) {
        match_t b = getBest(matchInfo1);
        if (b.valid) {
          muon.addStub(b.stubRef);
          if (b.isGlobal)
            muon.addMuonRef(b.muRef);
          quality += b.quality;
        }
      }
      if (!matchInfo2.empty()) {
        match_t b = getBest(matchInfo2);
        if (b.valid) {
          muon.addStub(b.stubRef);
          if (b.isGlobal)
            muon.addMuonRef(b.muRef);
          quality += b.quality;
        }
      }
      if (!matchInfo3.empty()) {
        match_t b = getBest(matchInfo3);
        if (b.valid) {
          muon.addStub(b.stubRef);
          if (b.isGlobal)
            muon.addMuonRef(b.muRef);
          quality += b.quality;
        }
      }
      if (!matchInfo4.empty()) {
        match_t b = getBest(matchInfo4);
        if (b.valid) {
          muon.addStub(b.stubRef);
          if (b.isGlobal)
            muon.addMuonRef(b.muRef);
          quality += b.quality;
        }
      }

      muon.setOfflineQuantities(track.offline_pt(), track.offline_eta(), track.offline_phi());
      muon.setTrkPtr(track.trkPtr());

      ap_uint<8> etaAddr = muon.eta() < 0 ? ap_uint<8>(-muon.eta() / 256) : ap_uint<8>((muon.eta()) / 256);
      ap_uint<8> ptAddr = muon.pt() > 4095 ? ap_uint<8>(15) : ap_uint<8>(muon.pt() / 256);
      ap_uint<8> addr = ptAddr | (etaAddr << 4);
      ap_uint<8> qualityCut = lt_tpsID[addr];

      if (quality >= qualityCut) {
        muon.setValid(true);
        muon.setQuality(quality + ptPenalty);
      } else {
        muon.setValid(false);
        muon.setQuality(0);
        muon.resetGlobal();
      }
      if (verbose_ == 1)
        muon.print();

      if (verbose_ == 1 && !rois.empty()) {  //patterns for HLS

        printf("TPS %d", track.trkPtr()->phiSector());
        track.printWord();

        for (uint i = 0; i < 16; ++i) {
          if (rois.size() > i) {
            rois[i].printROILine();
          } else {
            printf("%08x", 0);
            printf("%016lx", 0x1ff000000000000);
            printf("%016lx", 0x1ff000000000000);
            printf("%016lx", 0x1ff000000000000);
            printf("%016lx", 0x1ff000000000000);
            printf("%016lx", 0x1ff000000000000);
          }
        }
        muon.printWord();
        printf("\n");
      }
      return muon;
    }

    ap_uint<5> cleanMuon(const PreTrackMatchedMuon& mu, const PreTrackMatchedMuon& other, bool eq) {
      ap_uint<5> valid = 0;
      ap_uint<5> overlap = 0;
      if (mu.stubID0() != 511) {
        valid = valid | 0x1;
        if (mu.stubID0() == other.stubID0())
          overlap = overlap | 0x1;
      }
      if (mu.stubID1() != 511) {
        valid = valid | 0x2;
        if (mu.stubID1() == other.stubID1())
          overlap = overlap | 0x2;
      }
      if (mu.stubID2() != 511) {
        valid = valid | 0x4;
        if (mu.stubID2() == other.stubID2())
          overlap = overlap | 0x4;
      }
      if (mu.stubID3() != 511) {
        valid = valid | 0x8;
        if (mu.stubID3() == other.stubID3())
          overlap = overlap | 0x8;
      }
      if (mu.stubID4() != 511) {
        valid = valid | 0x10;
        if (mu.stubID4() == other.stubID4())
          overlap = overlap | 0x10;
      }

      if (((mu.quality() < other.quality()) && (!eq)) || ((mu.quality() <= other.quality()) && (eq)))
        return valid & (~overlap);
      else
        return valid;
    }

    std::vector<PreTrackMatchedMuon> clean(std::vector<PreTrackMatchedMuon>& muons) {
      std::vector<PreTrackMatchedMuon> out;
      if (muons.empty())
        return out;
      if (verbose_ == 1) {
        printf("-----Cleaning Up Muons in the same Nonant\n");
        printf("Before:\n");
      }
      for (uint i = 0; i < muons.size(); ++i) {
        if (verbose_ == 1)
          muons[i].print();

        ap_uint<5> mask = 0x1f;
        for (uint j = 0; j < muons.size(); ++j) {
          if (i == j)
            continue;
          mask = mask & cleanMuon(muons[i], muons[j], false);
        }
        if (mask) {
          if (verbose_ == 1)
            printf("kept\n");
          out.push_back(muons[i]);
        } else {
          if (verbose_ == 1)
            printf("discarded\n");
        }
      }
      return out;
    }
  };
}  // namespace Phase2L1GMT

#endif
