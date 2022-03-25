#ifndef DataFormatsL1TMuonPhase2_TrackerMuon_h
#define DataFormatsL1TMuonPhase2_TrackerMuon_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1t {

  class TrackerMuon;

  typedef std::vector<TrackerMuon> TrackerMuonCollection;
  typedef edm::Ref<TrackerMuonCollection> TrackerMuonRef;
  typedef std::vector<edm::Ref<TrackerMuonCollection> > TrackerMuonRefVector;

  class TrackerMuon : public L1Candidate {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef std::vector<L1TTTrackType> L1TTTrackCollection;

    TrackerMuon();

    TrackerMuon(
        const edm::Ptr<L1TTTrackType>& trk, bool charge, uint pt, int eta, int phi, int z0, int d0, uint quality);

    ~TrackerMuon() override;

    const edm::Ptr<L1TTTrackType>& trkPtr() const { return trkPtr_; }
    const edm::Ref<l1t::RegionalMuonCandBxCollection>& muonRef() const { return muRef_; }

    const bool hwCharge() const { return hwCharge_; }
    const int hwZ0() const { return hwZ0_; }
    const int hwD0() const { return hwD0_; }
    const int hwIsoSum() const { return hwIsoSum_; }
    const int hwIsoSumAp() const { return hwIsoSumAp_; }
    const uint hwBeta() const { return hwBeta_; }
    void setBeta(uint beta) { hwBeta_ = beta; }
    void setMuonRef(const edm::Ref<l1t::RegionalMuonCandBxCollection>& p) { muRef_ = p; }
    void setHwIsoSum(int isoSum) { hwIsoSum_ = isoSum; }
    void setHwIsoSumAp(int isoSum) { hwIsoSumAp_ = isoSum; }

    const std::array<uint64_t, 2> word() const { return word_; }
    void setWord(std::array<uint64_t, 2> word) { word_ = word; }
    void print() const;
    const MuonStubRefVector stubs() const { return stubs_; }
    void addStub(const MuonStubRef& stub) { stubs_.push_back(stub); }

    bool operator<(const TrackerMuon& other) const { return (hwPt() < other.hwPt()); }
    bool operator>(const TrackerMuon& other) const { return (hwPt() > other.hwPt()); }

  private:
    // used for the Naive producer
    edm::Ptr<L1TTTrackType> trkPtr_;
    bool hwCharge_;
    int hwZ0_;
    int hwD0_;
    uint hwBeta_;
    // The tracker muon is encoded in 96 bits as a 2-element array of uint64_t
    std::array<uint64_t, 2> word_ = {{0, 0}};
    //Store the eneryg sum for isolation
    int hwIsoSum_;
    //Store the eneryg sum for isolation with ap_type
    int hwIsoSumAp_;

    edm::Ref<l1t::RegionalMuonCandBxCollection> muRef_;
    MuonStubRefVector stubs_;
  };
}  // namespace l1t

#endif
