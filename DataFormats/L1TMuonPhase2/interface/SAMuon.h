#ifndef DataFormatsL1TMuonPhase2_SAMuon_h
#define DataFormatsL1TMuonPhase2_SAMuon_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
