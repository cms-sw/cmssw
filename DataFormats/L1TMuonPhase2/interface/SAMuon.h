#ifndef DataFormatsL1TMuonPhase2_SAMuon_h
#define DataFormatsL1TMuonPhase2_SAMuon_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"

namespace l1t {

  class SAMuon;

  typedef std::vector<SAMuon> SAMuonCollection;
  typedef edm::Ref<SAMuonCollection> SAMuonRef;
  typedef std::vector<edm::Ref<SAMuonCollection> > SAMuonRefVector;

  class SAMuon : public L1Candidate {
  public:
    SAMuon();

    SAMuon(const l1t::Muon& mu, bool charge, uint pt, int eta, int phi, int z0, int d0, uint quality);

    ~SAMuon() override;

    const bool hwCharge() const { return hwCharge_; }
    const int hwZ0() const { return hwZ0_; }
    const int hwD0() const { return hwD0_; }
    const uint hwBeta() const { return hwBeta_; }
    void setBeta(uint beta) { hwBeta_ = beta; }

    // For GT, returning ap_ type
    const Phase2L1GMT::valid_sa_t apValid() const { return Phase2L1GMT::valid_sa_t(hwPt() > 0); };
    const Phase2L1GMT::pt_sa_t apPt() const { return Phase2L1GMT::pt_sa_t(hwPt()); };
    const Phase2L1GMT::phi_sa_t apPhi() const { return Phase2L1GMT::phi_sa_t(hwPhi()); };
    const Phase2L1GMT::eta_sa_t apEta() const { return Phase2L1GMT::eta_sa_t(hwEta()); };
    const Phase2L1GMT::z0_sa_t apZ0() const { return Phase2L1GMT::z0_sa_t(hwZ0()); };
    const Phase2L1GMT::d0_sa_t apD0() const { return Phase2L1GMT::d0_sa_t(hwD0()); };
    const Phase2L1GMT::q_sa_t apCharge() const { return Phase2L1GMT::q_sa_t(hwCharge()); };
    const Phase2L1GMT::qual_sa_t apQual() const { return Phase2L1GMT::qual_sa_t(hwQual()); };

    // For HLT
    const double phZ0() const { return Phase2L1GMT::LSBSAz0 * hwZ0(); }
    const double phD0() const { return Phase2L1GMT::LSBSAd0 * hwD0(); }
    const double phPt() const { return Phase2L1GMT::LSBpt * hwPt(); }
    const double phEta() const { return Phase2L1GMT::LSBeta * hwEta(); }
    const double phPhi() const { return Phase2L1GMT::LSBphi * hwPhi(); }
    const int phCharge() const { return pow(-1, hwCharge()); }

    const uint64_t word() const { return word_; }
    void setWord(uint64_t word) { word_ = word; }
    void print() const;

    bool operator<(const SAMuon& other) const {
      if (hwPt() == other.hwPt())
        return (hwEta() < other.hwEta());
      else
        return (hwPt() < other.hwPt());
    }
    bool operator>(const SAMuon& other) const {
      if (hwPt() == other.hwPt())
        return (hwEta() > other.hwEta());
      else
        return (hwPt() > other.hwPt());
    }

  private:
    bool hwCharge_;
    int hwZ0_;
    int hwD0_;
    uint hwBeta_;
    uint64_t word_;
  };
}  // namespace l1t

#endif
