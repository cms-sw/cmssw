#ifndef DataFormats_L1TParticleFlow_PFCandidate_h
#define DataFormats_L1TParticleFlow_PFCandidate_h

#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"

namespace l1t {

  class PFCandidate : public L1Candidate {
  public:
    typedef edm::Ptr<l1t::Muon> MuonRef;
    enum ParticleType { ChargedHadron = 0, Electron = 1, NeutralHadron = 2, Photon = 3, Muon = 4 };

    PFCandidate() {}
    PFCandidate(ParticleType kind,
                int charge,
                const LorentzVector& p,
                float puppiWeight = -1,
                int hwpt = 0,
                int hweta = 0,
                int hwphi = 0)
        : PFCandidate(kind, charge, PolarLorentzVector(p), puppiWeight, hwpt, hweta, hwphi) {}
    PFCandidate(ParticleType kind,
                int charge,
                const PolarLorentzVector& p,
                float puppiWeight = -1,
                int hwpt = 0,
                int hweta = 0,
                int hwphi = 0);

    ParticleType id() const { return ParticleType(hwQual()); }

    const PFTrackRef& pfTrack() const { return trackRef_; }
    void setPFTrack(const PFTrackRef& ref) { trackRef_ = ref; }

    const PFClusterRef& pfCluster() const { return clusterRef_; }
    void setPFCluster(const PFClusterRef& ref) { clusterRef_ = ref; }

    const MuonRef& muon() const { return muonRef_; }
    void setMuon(const MuonRef& ref) { muonRef_ = ref; }

    /// PUPPI weight (-1 if not available)
    float puppiWeight() const { return puppiWeight_; }

  private:
    PFClusterRef clusterRef_;
    PFTrackRef trackRef_;
    MuonRef muonRef_;
    float puppiWeight_;

    void setPdgIdFromParticleType(int charge, ParticleType kind);
  };

  typedef std::vector<l1t::PFCandidate> PFCandidateCollection;
  typedef edm::Ref<l1t::PFCandidateCollection> PFCandidateRef;
  typedef edm::RefVector<l1t::PFCandidateCollection> PFCandidateRefVector;
}  // namespace l1t
#endif
