#ifndef PHASE2GMT_PRETRACKMATCHEDMUON
#define PHASE2GMT_PRETRACKMATCHEDMUON

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"

#include <vector>

namespace Phase2L1GMT {

  class PreTrackMatchedMuon {
  public:
    PreTrackMatchedMuon(const uint& charge,
                        const uint& pt,
                        const int& eta,
                        const int& phi,
                        const int& z0,
                        const int& d0,
                        const uint& beta = 15)
        : charge_(charge),
          pt_(pt),
          eta_(eta),
          phi_(phi),
          z0_(z0),
          d0_(d0),
          beta_(beta),
          isGlobal_(false),
          quality_(0),
          stubID0_(511),
          stubID1_(511),
          stubID2_(511),
          stubID3_(511),
          stubID4_(511),
          valid_(false) {}

    const uint charge() const { return charge_; }
    const uint pt() const { return pt_; }
    const int eta() const { return eta_; }
    const int phi() const { return phi_; }
    const int z0() const { return z0_; }
    const int d0() const { return d0_; }
    const uint beta() const { return beta_; }

    bool isGlobalMuon() const { return isGlobal_; }
    const int quality() const { return quality_; }
    const int offline_pt() const { return offline_pt_; }
    const float offline_eta() const { return offline_eta_; }
    const float offline_phi() const { return offline_phi_; }

    const uint stubID0() const { return stubID0_; }
    const uint stubID1() const { return stubID1_; }
    const uint stubID2() const { return stubID2_; }
    const uint stubID3() const { return stubID3_; }
    const uint stubID4() const { return stubID4_; }
    bool valid() const { return valid_; }

    void setQuality(uint quality) { quality_ = quality; }
    void setValid(bool v) { valid_ = v; }

    void setOfflineQuantities(float pt, float eta, float phi) {
      offline_pt_ = pt;
      offline_eta_ = eta;
      offline_phi_ = phi;
    }

    void addMuonRef(const l1t::RegionalMuonCandRef& ref) {
      muRef_.push_back(ref);
      isGlobal_ = true;
    }

    void resetGlobal() { isGlobal_ = false; }

    const std::vector<l1t::RegionalMuonCandRef>& muonRef() const { return muRef_; }
    void addStub(const l1t::MuonStubRef& stub) {
      stubs_.push_back(stub);
      if (stub->tfLayer() == 0)
        stubID0_ = stub->id();
      else if (stub->tfLayer() == 1)
        stubID1_ = stub->id();
      else if (stub->tfLayer() == 2)
        stubID2_ = stub->id();
      else if (stub->tfLayer() == 3)
        stubID3_ = stub->id();
      else if (stub->tfLayer() == 4)
        stubID4_ = stub->id();
    }

    const l1t::MuonStubRefVector& stubs() const { return stubs_; }

    void setTrkPtr(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& trkPtr) { trkPtr_ = trkPtr; }

    const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > trkPtr() const { return trkPtr_; }

    void print() const {
      LogDebug("PreTrackMatchedMuon") << "preconstructed muon : charge=" << charge_ << " pt=" << offline_pt_ << ","
                                      << pt_ << " eta=" << offline_eta_ << "," << eta_ << " phi=" << offline_phi_ << ","
                                      << phi_ << " z0=" << z0_ << " d0=" << d0_ << " quality=" << quality_
                                      << " isGlobal=" << isGlobal_ << " valid=" << valid_ << " stubs: " << stubID0_
                                      << " " << stubID1_ << " " << stubID2_ << " " << stubID3_ << " " << stubID4_;
    }

    uint64_t lsb() const {
      uint64_t w = charge_ & 0x1;
      w = w | (twos_complement(pt_, BITSPT) << 1);
      w = w | (twos_complement(phi_, BITSPHI) << (BITSPT + 1));
      w = w | (twos_complement(eta_, BITSETA) << (BITSPHI + BITSPT + 1));
      w = w | (twos_complement(z0_, BITSZ0) << (BITSETA + BITSPHI + BITSPT + 1));
      w = w | (twos_complement(d0_, BITSD0) << (BITSZ0 + BITSETA + BITSPHI + BITSPT + 1));
      return w;
    }

    uint64_t msb() const {
      uint64_t w2 = 0;
      w2 = twos_complement(stubID0_, BITSSTUBID);
      w2 = w2 | (twos_complement(stubID1_, BITSSTUBID) << BITSSTUBID);
      w2 = w2 | (twos_complement(stubID2_, BITSSTUBID) << (2 * BITSSTUBID));
      w2 = w2 | (twos_complement(stubID3_, BITSSTUBID) << (3 * BITSSTUBID));
      w2 = w2 | (twos_complement(stubID4_, BITSSTUBID) << (4 * BITSSTUBID));
      w2 = w2 | (twos_complement(isGlobal_, 1) << (5 * BITSSTUBID));
      w2 = w2 | (twos_complement(beta_, BITSMUONBETA) << (5 * BITSSTUBID + 1));
      w2 = w2 | (twos_complement(quality_, BITSMATCHQUALITY) << (BITSMUONBETA + 5 * BITSSTUBID + 1));
      w2 = w2 | (twos_complement(valid_, 1) << (BITSMATCHQUALITY + BITSMUONBETA + 5 * BITSSTUBID + 1));
      return w2;
    }

    void printWord() const {
      LogDebug("PreTrackMatchedMuon") << "PreTrackMatchedMuon : word=" << std::setfill('0') << std::setw(16) << std::hex
                                      << (long long unsigned int)(msb() >> 2) << std::setfill('0') << std::setw(16)
                                      << std::hex
                                      << (long long unsigned int)((lsb() | (msb() << 62)) & 0xffffffffffffffff);
    }

  private:
    uint charge_;
    uint pt_;
    int eta_;
    int phi_;
    int z0_;
    int d0_;
    uint beta_;
    bool isGlobal_;
    uint quality_;
    float offline_pt_;
    float offline_eta_;
    float offline_phi_;
    uint stubID0_;
    uint stubID1_;
    uint stubID2_;
    uint stubID3_;
    uint stubID4_;
    bool valid_;
    l1t::MuonStubRefVector stubs_;
    std::vector<l1t::RegionalMuonCandRef> muRef_;
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > trkPtr_;
  };
}  // namespace Phase2L1GMT

#endif
