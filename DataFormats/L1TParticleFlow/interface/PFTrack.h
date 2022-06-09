#ifndef DataFormats_L1TParticleFlow_PFTrack_h
#define DataFormats_L1TParticleFlow_PFTrack_h

#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/Ref.h"

namespace l1t {

  class PFTrack : public L1Candidate {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef edm::Ref<std::vector<L1TTTrackType>> TrackRef;

    PFTrack() {}
    PFTrack(int charge,
            const reco::Particle::LorentzVector& p4,
            const reco::Particle::Point& vtx,
            const TrackRef& tkPtr,
            int nPar,
            float caloEta,
            float caloPhi,
            float trkPtError = -1,
            float caloPtError = -1,
            int quality = 1,
            bool isMuon = false,
            int hwpt = 0,
            int hweta = 0,
            int hwphi = 0)
        : L1Candidate(p4, hwpt, hweta, hwphi, quality),
          trackRef_(tkPtr),
          caloEta_(caloEta),
          caloPhi_(caloPhi),
          trkPtError_(trkPtError),
          caloPtError_(caloPtError),
          isMuon_(isMuon),
          nPar_(nPar),
          trackWord_(*tkPtr) {
      setCharge(charge);
      setVertex(vtx);
    }

    const TrackRef& track() const { return trackRef_; }
    void setTrack(const TrackRef& ref) { trackRef_ = ref; }

    /// eta coordinate propagated at the calorimeter surface used for track-cluster matching
    float caloEta() const { return caloEta_; }
    /// phi coordinate propagated at the calorimeter surface used for track-cluster matching
    float caloPhi() const { return caloPhi_; }
    void setCaloEtaPhi(float eta, float phi) {
      caloEta_ = eta;
      caloPhi_ = phi;
    }

    /// uncertainty on track pt
    float trkPtError() const { return trkPtError_; }
    void setTrkPtError(float ptErr) { trkPtError_ = ptErr; }

    /// uncertainty on calorimetric response for a hadron with pt equal to this track's pt
    float caloPtError() const { return caloPtError_; }
    void setCaloPtError(float ptErr) { caloPtError_ = ptErr; }

    bool isMuon() const override { return isMuon_; }
    void setIsMuon(bool isMuon) { isMuon_ = isMuon; }

    int quality() const { return hwQual(); }
    void setQuality(int quality) { setHwQual(quality); }

    unsigned int nPar() const { return nPar_; }
    unsigned int nStubs() const { return track()->getStubRefs().size(); }
    float normalizedChi2() const { return track()->chi2Red(); }
    float chi2() const { return track()->chi2(); }

    const TTTrack_TrackWord& trackWord() const { return trackWord_; }
    TTTrack_TrackWord& trackWord() { return trackWord_; }

  private:
    TrackRef trackRef_;
    float caloEta_, caloPhi_;
    float trkPtError_;
    float caloPtError_;
    bool isMuon_;
    unsigned int nPar_;
    TTTrack_TrackWord trackWord_;
  };

  typedef std::vector<l1t::PFTrack> PFTrackCollection;
  typedef edm::Ref<l1t::PFTrackCollection> PFTrackRef;
  typedef edm::RefVector<l1t::PFTrackCollection> PFTrackRefVector;
  typedef std::vector<l1t::PFTrackRef> PFTrackVectorRef;
}  // namespace l1t
#endif
