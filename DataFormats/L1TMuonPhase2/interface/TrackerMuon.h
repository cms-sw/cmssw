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
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"

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
    const std::vector<l1t::RegionalMuonCandRef>& muonRef() const { return muRef_; }

    const bool hwCharge() const { return hwCharge_; }
    const int hwZ0() const { return hwZ0_; }
    const int hwD0() const { return hwD0_; }
    const int hwIsoSum() const { return hwIsoSum_; }
    const int hwIsoSumAp() const { return hwIsoSumAp_; }
    const uint hwBeta() const { return hwBeta_; }
    void setBeta(uint beta) { hwBeta_ = beta; }
    void setMuonRef(const std::vector<l1t::RegionalMuonCandRef>& p) { muRef_ = p; }
    void setHwIsoSum(int isoSum) { hwIsoSum_ = isoSum; }
    void setHwIsoSumAp(int isoSum) { hwIsoSumAp_ = isoSum; }

    // For GT, returning ap_ type
    const Phase2L1GMT::valid_gt_t apValid() const { return Phase2L1GMT::valid_gt_t(hwPt() > 0); };
    const Phase2L1GMT::pt_gt_t apPt() const { return Phase2L1GMT::pt_gt_t(hwPt()); };
    const Phase2L1GMT::phi_gt_t apPhi() const { return Phase2L1GMT::phi_gt_t(hwPhi()); };
    const Phase2L1GMT::eta_gt_t apEta() const { return Phase2L1GMT::eta_gt_t(hwEta()); };
    const Phase2L1GMT::z0_gt_t apZ0() const { return Phase2L1GMT::z0_gt_t(hwZ0()); };
    const Phase2L1GMT::d0_gt_t apD0() const { return Phase2L1GMT::d0_gt_t(hwD0()); };
    const Phase2L1GMT::q_gt_t apCharge() const { return Phase2L1GMT::q_gt_t(hwCharge()); };
    const Phase2L1GMT::qual_gt_t apQual() const { return Phase2L1GMT::qual_gt_t(hwQual()); };
    const Phase2L1GMT::iso_gt_t apIso() const { return Phase2L1GMT::iso_gt_t(hwIso()); };
    const Phase2L1GMT::beta_gt_t apBeta() const { return Phase2L1GMT::beta_gt_t(hwBeta()); };

    // For HLT
    const double phZ0() const { return Phase2L1GMT::LSBGTz0 * hwZ0(); }
    const double phD0() const { return Phase2L1GMT::LSBGTd0 * hwD0(); }
    const double phPt() const { return Phase2L1GMT::LSBpt * hwPt(); }
    const double phEta() const { return Phase2L1GMT::LSBeta * hwEta(); }
    const double phPhi() const { return Phase2L1GMT::LSBphi * hwPhi(); }
    const int phCharge() const { return pow(-1, hwCharge()); }

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

    std::vector<l1t::RegionalMuonCandRef> muRef_;
    MuonStubRefVector stubs_;
  };
}  // namespace l1t

#endif
