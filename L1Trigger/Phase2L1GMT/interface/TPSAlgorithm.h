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
#include <fstream>
#include <iostream>

namespace Phase2L1GMT {

  const unsigned int PHIDIVIDER = 1 << (BITSPHI - BITSSTUBCOORD);
  const unsigned int ETADIVIDER = 1 << (BITSETA - BITSSTUBETA);
  const unsigned int BITSPROP = BITSPHI - 2;
  const unsigned int PROPMAX = ~ap_uint<BITSPROP>(0);

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
    ap_uint<2> valid;
    bool isGlobal;
    l1t::SAMuonRef muRef;
    l1t::MuonStubRef stubRef;

  } match_t;

  class TPSAlgorithm {
  public:
    TPSAlgorithm(const edm::ParameterSet& iConfig);
    ~TPSAlgorithm();

    std::vector<PreTrackMatchedMuon> processNonant(const std::vector<ConvertedTTTrack>& convertedTracks,
                                                   const l1t::MuonStubRefVector& stubs);

    std::vector<PreTrackMatchedMuon> cleanNeighbor(const std::vector<PreTrackMatchedMuon>& muons,
                                                   const std::vector<PreTrackMatchedMuon>& muonsPrevious,
                                                   const std::vector<PreTrackMatchedMuon>& muonsNext,
                                                   bool equality);
    std::vector<l1t::TrackerMuon> convert(std::vector<PreTrackMatchedMuon>& muons, uint maximum);
    bool outputGT(std::vector<l1t::TrackerMuon>& muons);
    void SetQualityBits(std::vector<l1t::TrackerMuon>& muons);
    std::vector<l1t::TrackerMuon> sort(std::vector<l1t::TrackerMuon>& muons, uint maximum);

  private:
    int verbose_;
    propagation_t propagate(const ConvertedTTTrack& track, uint layer);
    ap_uint<BITSSIGMAETA + 1> deltaEta(const ap_int<BITSSTUBETA>& eta1, const ap_int<BITSSTUBETA>& eta2);
    ap_uint<BITSSIGMACOORD + 1> deltaCoord(const ap_int<BITSSTUBCOORD>& phi1, const ap_int<BITSSTUBCOORD>& phi2);
    match_t match(const propagation_t prop, const l1t::MuonStubRef& stub, uint trackID);
    match_t propagateAndMatch(const ConvertedTTTrack& track, const l1t::MuonStubRef& stub, uint trackID);
    match_t getBest(const std::vector<match_t> matches);
    PreTrackMatchedMuon processTrack(const ConvertedTTTrack&, const l1t::MuonStubRefVector&);
    ap_uint<5> cleanMuon(const PreTrackMatchedMuon& mu, const PreTrackMatchedMuon& other, bool eq);
    void matchingInfos(std::vector<match_t> matchInfo, PreTrackMatchedMuon& muon, ap_uint<BITSMATCHQUALITY>& quality);
    std::vector<PreTrackMatchedMuon> clean(std::vector<PreTrackMatchedMuon>& muons);
  };
}  // namespace Phase2L1GMT

#endif
