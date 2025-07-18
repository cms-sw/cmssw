#ifndef PHASE2GMT_CONEVRTEDTTRACK
#define PHASE2GMT_CONEVRTEDTTRACK
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <ap_int.h>

namespace Phase2L1GMT {

  class ConvertedTTTrack {
  public:
    ConvertedTTTrack(const ap_uint<1>& charge,
                     const ap_int<BITSTTCURV>& curvature,
                     const ap_int<BITSETA - 1>& abseta,
                     const ap_uint<BITSPT>& pt,
                     const ap_int<BITSETA>& eta,
                     const ap_int<BITSPHI>& phi,
                     const ap_int<BITSZ0>& z0,
                     const ap_int<BITSD0>& d0,
                     const ap_uint<1>& quality,
                     const ap_uint<96>& word)
        : charge_(charge),
          curvature_(curvature),
          abseta_(abseta),
          pt_(pt),
          eta_(eta),
          phi_(phi),
          z0_(z0),
          d0_(d0),
          quality_(quality),
          word_(word) {}

    const ap_uint<1> charge() const { return charge_; }

    const ap_int<BITSTTCURV> curvature() const { return curvature_; }
    const ap_uint<BITSETA - 1> abseta() const { return abseta_; }

    const ap_uint<BITSPT> pt() const { return pt_; }

    const ap_int<BITSETA> eta() const { return eta_; }
    const ap_int<BITSPHI> phi() const { return phi_; }

    void setPhi(ap_int<BITSPHI> phi) { phi_ = phi; }

    const ap_int<BITSZ0> z0() const { return z0_; }
    const ap_int<BITSD0> d0() const { return d0_; }
    const ap_uint<1> quality() const { return quality_; }
    const float offline_pt() const { return offline_pt_; }
    const float offline_eta() const { return offline_eta_; }
    const float offline_phi() const { return offline_phi_; }

    const ap_uint<96>& word() const { return word_; }
    void setOfflineQuantities(float pt, float eta, float phi) {
      offline_pt_ = pt;
      offline_eta_ = eta;
      offline_phi_ = phi;
    }

    void print() const {
      LogDebug("ConvertedTTTrack") << "converted track : charge=" << charge_ << " curvature=" << curvature_
                                   << " pt=" << offline_pt_ << "," << pt_ << " eta=" << offline_eta_ << "," << eta_
                                   << " phi=" << offline_phi_ << "," << phi_ << " z0=" << z0_ << " d0=" << d0_
                                   << " quality=" << quality_;
    }

    void printWord() const {
      LogDebug("ConvertedTTTrack") << "converted track : word=" << std::setfill('0') << std::setw(8) << std::hex
                                   << (long long unsigned int)((word_ >> 64).to_uint64()) << std::setfill('0')
                                   << std::setw(16) << std::hex
                                   << (long long unsigned int)((word_ & 0xffffffffffffffff).to_uint64());
    }

    void setTrkPtr(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& trkPtr) { trkPtr_ = trkPtr; }

    const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > trkPtr() const { return trkPtr_; }

  private:
    ap_uint<1> charge_;
    ap_int<BITSTTCURV> curvature_;
    ap_uint<BITSETA - 1> abseta_;
    ap_uint<BITSPT> pt_;
    ap_int<BITSETA> eta_;
    ap_int<BITSPHI> phi_;
    ap_int<BITSZ0> z0_;
    ap_int<BITSD0> d0_;
    ap_uint<1> quality_;
    float offline_pt_;
    float offline_eta_;
    float offline_phi_;
    ap_uint<96> word_;

    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > trkPtr_;
  };
}  // namespace Phase2L1GMT

#endif
