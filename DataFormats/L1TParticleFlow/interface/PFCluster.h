#ifndef DataFormats_L1TParticleFlow_PFCluster_h
#define DataFormats_L1TParticleFlow_PFCluster_h

#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

namespace l1t {

  class PFCluster : public L1Candidate {
  public:
    /// constituent information. note that this is not going to be available in the hardware!
    typedef std::pair<edm::Ptr<l1t::L1Candidate>, float> ConstituentAndFraction;
    typedef std::vector<ConstituentAndFraction> ConstituentsAndFractions;

    PFCluster() {}
    PFCluster(float pt,
              float eta,
              float phi,
              float hOverE = 0,
              bool isEM = false,
              float ptError = 0,
              int hwpt = 0,
              int hweta = 0,
              int hwphi = 0,
              float absZBarycenter = 0.,
              float sigmaRR = 0.)
        : L1Candidate(PolarLorentzVector(pt, eta, phi, 0), hwpt, hweta, hwphi, /*hwQuality=*/isEM ? 1 : 0),
          hOverE_(hOverE),
          ptError_(ptError),
          absZBarycenter_(absZBarycenter),
          sigmaRR_(sigmaRR) {
      setPdgId(isEM ? 22 : 130);  // photon : non-photon(K0)
    }
    PFCluster(
        const LorentzVector& p4, float hOverE, bool isEM, float ptError = 0, int hwpt = 0, int hweta = 0, int hwphi = 0)
        : L1Candidate(p4, hwpt, hweta, hwphi, /*hwQuality=*/isEM ? 1 : 0), hOverE_(hOverE), ptError_(ptError) {
      setPdgId(isEM ? 22 : 130);  // photon : non-photon(K0)
    }

    float hOverE() const { return hOverE_; }
    void setHOverE(float hOverE) { hOverE_ = hOverE; }

    void setSigmaRR(float sigmaRR) { sigmaRR_ = sigmaRR; }
    float absZBarycenter() const { return absZBarycenter_; }

    void setAbsZBarycenter(float absZBarycenter) { absZBarycenter_ = absZBarycenter; }
    float sigmaRR() const { return sigmaRR_; }

    float emEt() const {
      if (hOverE_ == -1)
        return 0;
      return pt() / (1 + hOverE_);
    }

    // change the pt. H/E also recalculated to keep emEt constant
    void calibratePt(float newpt, float preserveEmEt = true);

    /// constituent information. note that this is not going to be available in the hardware!
    const ConstituentsAndFractions& constituentsAndFractions() const { return constituents_; }
    /// adds a candidate to this cluster; note that this only records the information, it's up to you to also set the 4-vector appropriately
    void addConstituent(const edm::Ptr<l1t::L1Candidate>& cand, float fraction = 1.0) {
      constituents_.emplace_back(cand, fraction);
    }

    float ptError() const { return ptError_; }
    void setPtError(float ptError) { ptError_ = ptError; }

    bool isEM() const { return hwQual(); }
    void setIsEM(bool isEM) { setHwQual(isEM); }
    unsigned int hwEmID() const { return hwQual(); }

    float egVsPionMVAOut() const { return egVsPionMVAOut_; }
    void setEgVsPionMVAOut(float egVsPionMVAOut) { egVsPionMVAOut_ = egVsPionMVAOut; }

    float egVsPUMVAOut() const { return egVsPUMVAOut_; }
    void setEgVsPUMVAOut(float egVsPUMVAOut) { egVsPUMVAOut_ = egVsPUMVAOut; }

  private:
    float hOverE_, ptError_, egVsPionMVAOut_, egVsPUMVAOut_;
    // HGC dedicated quantities (0ed by default)
    float absZBarycenter_, sigmaRR_;

    ConstituentsAndFractions constituents_;
  };

  typedef std::vector<l1t::PFCluster> PFClusterCollection;
  typedef edm::Ref<l1t::PFClusterCollection> PFClusterRef;
}  // namespace l1t
#endif
