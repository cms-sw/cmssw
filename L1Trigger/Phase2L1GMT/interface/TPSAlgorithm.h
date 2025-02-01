#ifndef L1Trigger_Phase2GMT_TPSAlgorithm_h
#define L1Trigger_Phase2GMT_TPSAlgorithm_h

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/Phase2L1GMT/interface/ConvertedTTTrack.h"
#include "L1Trigger/Phase2L1GMT/interface/PreTrackMatchedMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/TPSLUTs.h"
#include <iostream>

namespace Phase2L1GMT {

  const unsigned int PHISHIFT = BITSPHI - BITSSTUBCOORD;
  const unsigned int ETADIVIDER = 1 << (BITSETA - BITSSTUBETA);
  const unsigned int BITSHIFTPROP1C1 = 9;
  const unsigned int BITSHIFTPROP2C1 = 23;
  const unsigned int BITSHIFTPROP3C1 = 22;
  const unsigned int BITSHIFTPROP1C2 = 9;
  const unsigned int BITSHIFTPROP2C2 = 24;
  const unsigned int BITSHIFTPROP3C2 = 19;
  const unsigned int BITSHIFTRES1 = 11;
  //for comparison with absK to see which functional form to propagate phi according to
  //coord1 k cutoff: 4096
  //coord2 k cutoffs: 1024, 7168, 4096, 2048, 4096
  const unsigned int BITSHIFTCURVSCALEC1 = 12;
  const unsigned int BITSHIFTCURVSCALEC2LEADS[5] = {10, 13, 12, 11, 12};
  const unsigned int BITSHIFTCURVSCALEC2CORRS[5] = {0,  10,  0,  0,  0};
  
  const unsigned int BITSPROP = BITSPHI - 2;
  const ap_uint<BITSPROP> PROPMAX = ~ap_uint<BITSPROP>(0);
  const ap_uint<BITSSIGMACOORD> SIGMAMAX = ~ap_uint<BITSSIGMACOORD>(0);
  const ap_uint<BITSSIGMACOORD> SIGMAMIN = 2;
  
  struct propagation_t {
    ap_int<BITSSTUBCOORD> coord1;
    ap_uint<BITSSIGMACOORD> sigma_coord1;
    ap_int<BITSSTUBCOORD> coord2;
    ap_uint<BITSSIGMACOORD> sigma_coord2;
    ap_int<BITSSTUBETA> eta;
    ap_uint<BITSSIGMAETA> sigma_eta1;
    ap_uint<BITSSIGMAETA> sigma_eta2;
    ap_uint<1> valid;
    ap_uint<1> is_barrel;
  };

  struct match_t {
    ap_uint<BITSMATCHQUALITY - 2> quality;
    ap_uint<BITSSTUBID> id;
    ap_uint<2> valid;
    bool isGlobal = false;
    l1t::SAMuonRef muRef;
    l1t::MuonStubRef stubRef;
  };

  class TPSAlgorithm {
  public:
    explicit TPSAlgorithm(const edm::ParameterSet& iConfig);
    TPSAlgorithm() = delete;
    ~TPSAlgorithm() = default;

    std::vector<PreTrackMatchedMuon> processNonant(const std::vector<ConvertedTTTrack>& convertedTracks,
                                                   const l1t::MuonStubRefVector& stubs) const;

    std::vector<PreTrackMatchedMuon> cleanNeighbor(const std::vector<PreTrackMatchedMuon>& muons,
                                                   const std::vector<PreTrackMatchedMuon>& muonsPrevious,
                                                   const std::vector<PreTrackMatchedMuon>& muonsNext,
                                                   bool equality) const;
    std::vector<l1t::TrackerMuon> convert(const std::vector<PreTrackMatchedMuon>& muons, uint maximum) const;
    bool outputGT(std::vector<l1t::TrackerMuon>& muons) const;
    void SetQualityBits(std::vector<l1t::TrackerMuon>& muons) const;
    std::vector<l1t::TrackerMuon> sort(std::vector<l1t::TrackerMuon>& muons, uint maximum) const;

  private:
    int verbose_;
    propagation_t propagate(const ConvertedTTTrack& track, uint layer) const;
    ap_uint<BITSSIGMAETA + 1> deltaEta(const ap_int<BITSSTUBETA>& eta1, const ap_int<BITSSTUBETA>& eta2) const;
    ap_uint<BITSSIGMACOORD + 1> deltaCoord(const ap_int<BITSSTUBCOORD>& phi1, const ap_int<BITSSTUBCOORD>& phi2) const;
    match_t match(const propagation_t prop, const l1t::MuonStubRef& stub, uint trackID) const;
    match_t propagateAndMatch(const ConvertedTTTrack& track, const l1t::MuonStubRef& stub, uint trackID) const;
    match_t getBest(const std::vector<match_t>& matches) const;
    PreTrackMatchedMuon processTrack(const ConvertedTTTrack&, const l1t::MuonStubRefVector&) const;
    ap_uint<5> cleanMuon(const PreTrackMatchedMuon& mu, const PreTrackMatchedMuon& other, bool eq) const;
    void matchingInfos(const std::vector<match_t>& matchInfo,
                       PreTrackMatchedMuon& muon,
                       ap_uint<BITSMATCHQUALITY>& quality) const;
    std::vector<PreTrackMatchedMuon> clean(const std::vector<PreTrackMatchedMuon>& muons) const;
  };
}  // namespace Phase2L1GMT

#endif
