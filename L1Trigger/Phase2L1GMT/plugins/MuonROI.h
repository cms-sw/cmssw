#ifndef PHASE2GMT_MUONROI
#define PHASE2GMT_MUONROI
#include <iosfwd>
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace Phase2L1GMT {

  class MuonROI {
  public:
    MuonROI(int bx, uint charge, uint pt, uint quality) : bx_(bx), charge_(charge), pt_(pt), quality_(quality) {}

    const int bx() const { return bx_; }

    const uint charge() const { return charge_; }

    const uint pt() const { return pt_; }
    const int quality() const { return quality_; }

    const float offline_pt() const { return offline_pt_; }

    void setOfflinePt(float pt) { offline_pt_ = pt; }

    void addStub(const l1t::MuonStubRef& stub) { stubs_.push_back(stub); }

    void setMuonRef(const l1t::RegionalMuonCandRef& ref) {
      muRef_ = ref;
      isGlobal_ = true;
    }
    bool isGlobalMuon() const { return isGlobal_; }

    const l1t::RegionalMuonCandRef& muonRef() const { return muRef_; }

    friend std::ostream& operator<<(std::ostream& s, const MuonROI& id) {
      s.setf(ios::right, ios::adjustfield);
      s << "ROI:"
        << " "
        << "BX: " << setw(5) << id.bx_ << " "
        << "charge:" << setw(5) << id.charge_ << " "
        << "pt:" << setw(5) << id.pt_ << " "
        << "quality:" << setw(5) << id.quality_ << " "
        << "offline pt:" << setw(5) << id.offline_pt_;
      return s;
    }

    const l1t::MuonStubRefVector& stubs() const { return stubs_; }

    ap_uint<64> stubWord(const l1t::MuonStubRef& stub) const {
      ap_uint<64> word = 0;
      word = word | twos_complement(stub->coord1(), BITSSTUBCOORD);
      word = word | (twos_complement(stub->coord2(), BITSSTUBCOORD) << BITSSTUBCOORD);
      word = word | (twos_complement(stub->eta1(), BITSSTUBETA) << (2 * BITSSTUBCOORD));
      word = word | (twos_complement(stub->eta2(), BITSSTUBETA) << (2 * BITSSTUBCOORD + BITSSTUBETA));
      word = word | (twos_complement(stub->quality(), BITSSTUBPHIQUALITY) << (2 * BITSSTUBCOORD + 2 * BITSSTUBETA));
      word = word | (twos_complement(stub->etaQuality(), BITSSTUBETAQUALITY)
                     << (2 * BITSSTUBCOORD + 2 * BITSSTUBETA + BITSSTUBPHIQUALITY));
      word = word | (twos_complement(stub->bxNum(), BITSSTUBTIME)
                     << (2 * BITSSTUBCOORD + 2 * BITSSTUBETA + BITSSTUBPHIQUALITY + BITSSTUBETAQUALITY));
      word = word | (twos_complement(stub->id(), BITSSTUBID)
                     << (2 * BITSSTUBCOORD + 2 * BITSSTUBETA + BITSSTUBPHIQUALITY + BITSSTUBETAQUALITY + BITSSTUBTIME));
      return word;
    }

    ap_uint<32> roiWord() const {
      ap_uint<32> word = 0;
      word = word | twos_complement(bx_, BITSMUONBX);
      word = word | (twos_complement(isGlobal_, 1) << (BITSMUONBX));
      word = word | (twos_complement(charge_, 1) << (BITSMUONBX + 1));
      word = word | (twos_complement(pt_, BITSPT) << (BITSMUONBX + 2));
      word = word | (twos_complement(quality_, BITSSTAMUONQUALITY) << (BITSMUONBX + 2 + BITSPT));
      return word;
    }

    void printROILine() const {
      ap_uint<64> s0 = 0x1ff000000000000;
      ap_uint<64> s1 = 0x1ff000000000000;
      ap_uint<64> s2 = 0x1ff000000000000;
      ap_uint<64> s3 = 0x1ff000000000000;
      ap_uint<64> s4 = 0x1ff000000000000;
      for (const auto& s : stubs_) {
        if (s->tfLayer() == 0)
          s0 = stubWord(s);
        if (s->tfLayer() == 1)
          s1 = stubWord(s);
        if (s->tfLayer() == 2)
          s2 = stubWord(s);
        if (s->tfLayer() == 3)
          s3 = stubWord(s);
        if (s->tfLayer() == 4)
          s4 = stubWord(s);
      }
      LogDebug("MuonROI") << "MuonROI " << std::setfill('0') << std::setw(8) << std::hex
                          << (long long unsigned int)(roiWord().to_uint64()) << std::setfill('0') << std::setw(16)
                          << std::hex << (long long unsigned int)(s4.to_uint64()) << std::setfill('0') << std::setw(16)
                          << std::hex << (long long unsigned int)(s3.to_uint64()) << std::setfill('0') << std::setw(16)
                          << std::hex << (long long unsigned int)(s2.to_uint64()) << std::setfill('0') << std::setw(16)
                          << std::hex << (long long unsigned int)(s1.to_uint64()) << std::setfill('0') << std::setw(16)
                          << std::hex << (long long unsigned int)(s0.to_uint64());
    }

  private:
    int bx_;
    uint charge_;
    uint pt_;
    uint quality_;
    bool isGlobal_;
    float offline_pt_;

    l1t::MuonStubRefVector stubs_;
    l1t::RegionalMuonCandRef muRef_;
  };
}  // namespace Phase2L1GMT

#endif
