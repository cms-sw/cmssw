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
                     const ap_uint<1>& quality)
        : charge_(charge),
          curvature_(curvature),
          abseta_(abseta),
          pt_(pt),
          eta_(eta),
          phi_(phi),
          z0_(z0),
          d0_(d0),
          quality_(quality) {}

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

    const ap_uint<67> word() const {
      ap_uint<67> w = 0;
      int bstart = 0;
      bstart = wordconcat<ap_uint<67>>(w, bstart, 1, 1); //valid bit is always on in emulator because invalid tracks are never created
      bstart = wordconcat<ap_uint<67>>(w, bstart, charge_, 1);
      bstart = wordconcat<ap_uint<67>>(w, bstart, pt_, BITSPT);
      bstart = wordconcat<ap_uint<67>>(w, bstart, phi_, BITSPHI);
      bstart = wordconcat<ap_uint<67>>(w, bstart, eta_, BITSETA);
      bstart = wordconcat<ap_uint<67>>(w, bstart, z0_, BITSZ0);
      bstart = wordconcat<ap_uint<67>>(w, bstart, d0_, BITSD0);
      bstart = wordconcat<ap_uint<67>>(w, bstart, quality_, 1);
      return w;
    }

    void setOfflineQuantities(float pt, float eta, float phi) {
      offline_pt_ = pt;
      offline_eta_ = eta;
      offline_phi_ = phi;
    }

    void print(std::string module="ConvertedTTTrack", uint spaces=0, bool label=true) const {
      std::string lab = "";
      lab.append(spaces, ' ');
      if (label)
        lab.append("converted track:    ");
      std::string chargeSign = (charge_ == 0) ? "+1" : "-1";
      edm::LogInfo(module) << lab
      	                   << "charge = " << chargeSign << " (" << charge_ << ")" << ",    "
                           << "pt = " << offline_pt_ << " (" << pt_ << ")" << ",    "
                           << "phi = " << offline_phi_ << " (" << phi_ << ")" << ",    "
		           << "eta = " << offline_eta_ << " (" << eta_ << ")" << ","
			   << "\n" << std::setfill(' ') << std::setw(lab.length()) << " "
			   << "z0 = " << z0_ << ",    " 
			   << "d0 = " << d0_ << ",    "
                           << "quality = " << quality_ << ",    "
			   << "(curvature = " << curvature_ << ")"
			   << std::flush;
    }
                           
    void printWord(std::string module="ConvertedTTTrack", uint spaces=0, bool label=true) const {
      std::string lab = "";
      lab.append(spaces, ' ');
      if (label)
        lab.append("converted track word = ");
      ap_uint<67> w = word();
      edm::LogInfo(module) << lab
	                   << std::setfill('0') << std::setw(1)  << std::hex << (long long unsigned int)((w >> 64).to_uint64()) 
			   << std::setfill('0') << std::setw(16) << std::hex << (long long unsigned int)((w & 0xffffffffffffffff).to_uint64())
			   << std::flush;
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

    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > trkPtr_;
  };
}  // namespace Phase2L1GMT

#endif
