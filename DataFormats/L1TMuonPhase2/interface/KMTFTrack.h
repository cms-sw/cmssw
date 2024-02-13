#ifndef DataFormats_L1TMuonPhase2_KMTFTrack_h
#define DataFormats_L1TMuonPhase2_KMTFTrack_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class KMTFTrack;
  typedef std::vector<KMTFTrack> KMTFTrackCollection;
  typedef BXVector<KMTFTrack> KMTFTrackBxCollection;

  class KMTFTrack : public reco::LeafCandidate {
  public:
    KMTFTrack()
        : reco::LeafCandidate(-1, reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          unconstrainedP4_(reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          covariance_(std::vector<double>(6, 0.0)),
          curvVertex_(0),
          ptC_(0),
          phiVertex_(0),
          dxy_(0),
          curvMuon_(0),
          ptU_(0),
          phiMuon_(0),
          phiBMuon_(0),
          curv_(0),
          phi_(0),
          phiB_(0),
          coarseEta_(0),
          approxPromptChi2_(0),
          approxPromptErrChi2_(0),
          approxDispChi2_(0),
          approxDispErrChi2_(0),
          hitPattern_(0),
          step_(1),
          sector_(0),
          wheel_(0),
          quality_(0),
          hasFineEta_(false),
          bx_(0),
          rankPrompt_(0),
          rankDisp_(0),
          idFlag_(0) {}

    ~KMTFTrack() = default;

    KMTFTrack(const l1t::MuonStubRef& seed, int phi, int phiB)
        : reco::LeafCandidate(-1, reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          unconstrainedP4_(reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          covariance_(std::vector<double>(6, 0.0)),
          curvVertex_(0),
          ptC_(0),
          phiVertex_(0),
          dxy_(0),
          curvMuon_(0),
          ptU_(0),
          phiMuon_(0),
          phiBMuon_(0),
          curv_(0),
          phi_(phi),
          phiB_(phiB),
          coarseEta_(0),
          approxPromptChi2_(0),
          approxPromptErrChi2_(0),
          approxDispChi2_(0),
          approxDispErrChi2_(0),
          hitPattern_(0),
          step_(seed->depthRegion()),
          sector_(seed->phiRegion()),
          wheel_(seed->etaRegion()),
          quality_(seed->quality()),
          hasFineEta_(false),
          bx_(seed->bxNum()),
          rankPrompt_(0),
          rankDisp_(0),
          idFlag_(0) {
      stubs_.push_back(seed);
      residuals_.push_back(0);
      residuals_.push_back(0);
      residuals_.push_back(0);
    }

    reco::LeafCandidate::PolarLorentzVector displacedP4() const { return unconstrainedP4_; }

    //unconstrained pt
    int ptDisplaced() const { return ptU_; }
    //unconstrained curvature at station 1
    int curvatureAtMuon() const { return curvMuon_; }
    //unconstrained phi at station 1
    int phiAtMuon() const { return phiMuon_; }
    //unconstrained phiB at station 1
    int phiBAtMuon() const { return phiBMuon_; }

    //constrained pt
    int ptPrompt() const { return ptC_; }
    //Constrained curvature at vertex
    int curvatureAtVertex() const { return curvVertex_; }
    //constrained phi at the vertex
    int phiAtVertex() const { return phiVertex_; }
    //Impact parameter as calculated from the muon track
    int dxy() const { return dxy_; }
    //Unconstrained curvature at the Muon systen
    int curvature() const { return curv_; }
    //Unconstrained phi at the Muon systen
    int positionAngle() const { return phi_; }
    //Unconstrained bending angle at the Muon systen
    int bendingAngle() const { return phiB_; }
    //Coarse eta caluclated only using phi segments
    int coarseEta() const { return coarseEta_; }
    //Approximate Chi2 metrics
    int approxPromptChi2() const { return approxPromptChi2_; }
    int approxPromptErrChi2() const { return approxPromptErrChi2_; }
    int approxDispChi2() const { return approxDispChi2_; }
    int approxDispErrChi2() const { return approxDispErrChi2_; }

    int hitPattern() const { return hitPattern_; }
    //step;
    int step() const { return step_; }
    //sector;
    int sector() const { return sector_; }
    //wheel
    int wheel() const { return wheel_; }
    //quality
    int quality() const { return quality_; }

    //fine eta
    int fineEta() const { return fineEta_; }
    bool hasFineEta() const { return hasFineEta_; }

    //BX
    int bx() const { return bx_; }

    //rank
    int rankPrompt() const { return rankPrompt_; }
    int rankDisp() const { return rankDisp_; }

    int id() const { return idFlag_; }

    //Associated stubs
    const l1t::MuonStubRefVector& stubs() const { return stubs_; }

    //get Kalman gain
    const std::vector<float>& kalmanGain(unsigned int step) const {
      switch (step) {
        case 3:
          return kalmanGain3_;
        case 2:
          return kalmanGain2_;
        case 1:
          return kalmanGain1_;
        case 0:
          return kalmanGain0_;
      }
      return kalmanGain0_;
    }

    //get covariance
    const std::vector<double>& covariance() const { return covariance_; }

    //get residual
    int residual(uint i) const { return residuals_[i]; }

    //check overlap
    bool overlapTrack(const KMTFTrack& other) const {
      for (const auto& s1 : stubs_) {
        for (const auto& s2 : other.stubs()) {
          if (s1->phiRegion() == s2->phiRegion() && s1->etaRegion() == s2->etaRegion() &&
              s1->depthRegion() == s2->depthRegion() && s1->id() == s2->id())
            return true;
        }
      }
      return false;
    }

    bool operator==(const KMTFTrack& t2) const {
      if (this->stubs().size() != t2.stubs().size())
        return false;
      for (unsigned int i = 0; i < this->stubs().size(); ++i) {
        const l1t::MuonStubRef& s1 = this->stubs()[i];
        const l1t::MuonStubRef& s2 = t2.stubs()[i];
        if (s1->phiRegion() != s2->phiRegion() || s1->etaRegion() != s2->etaRegion() ||
            s1->depthRegion() != s2->depthRegion() || s1->id() != s2->id() || s1->tfLayer() != s2->tfLayer())
          return false;
      }
      return true;
    }

    //Set coordinates general
    void setCoordinates(int step, int curv, int phi, int phiB) {
      step_ = step;
      curv_ = curv;
      phiB_ = phiB;
      phi_ = phi;
    }

    void setCoordinatesAtVertex(int curv, int phi, int dxy) {
      curvVertex_ = curv;
      phiVertex_ = phi;
      dxy_ = dxy;
    }

    void setCoordinatesAtMuon(int curv, int phi, int phiB) {
      curvMuon_ = curv;
      phiMuon_ = phi;
      phiBMuon_ = phiB;
    }

    void setPt(int ptC, int ptU) {
      ptC_ = ptC;
      ptU_ = ptU;
    }

    void setCoarseEta(int eta) { coarseEta_ = eta; }

    void setHitPattern(int pattern) { hitPattern_ = pattern; }

    void setApproxChi2(int chi, int chiErr, bool prompt) {
      if (prompt) {
        approxPromptChi2_ = chi;
        approxPromptErrChi2_ = chiErr;
      } else {
        approxDispChi2_ = chi;
        approxDispErrChi2_ = chiErr;
      }
    }

    void setPtEtaPhi(double pt, double eta, double phi) {
      PolarLorentzVector v(pt, eta, phi, 0.105);
      setP4(v);
    }
    void setPtEtaPhiDisplaced(double pt, double eta, double phi) {
      unconstrainedP4_.SetPt(pt);
      unconstrainedP4_.SetEta(eta);
      unconstrainedP4_.SetPhi(phi);
    }

    void addStub(const l1t::MuonStubRef& stub) {
      if (stub->quality() < quality_)
        quality_ = stub->quality();
      stubs_.push_back(stub);
    }

    void setStubs(const l1t::MuonStubRefVector& stubs) { stubs_ = stubs; }

    void setRank(int rank, bool vertex) {
      if (vertex)
        rankPrompt_ = rank;
      else
        rankDisp_ = rank;
    }

    void setIDFlag(bool passPrompt, bool passDisp) {
      unsigned p0 = 0;
      unsigned p1 = 0;

      if (passPrompt)
        p0 = 1;
      if (passDisp)
        p1 = 2;

      idFlag_ = p0 | p1;
    }

    void setKalmanGain(
        unsigned int step, unsigned int K, float a1, float a2, float a3 = 0, float a4 = 0, float a5 = 0, float a6 = 0) {
      switch (step) {
        case 3:
          kalmanGain3_.push_back(K);
          kalmanGain3_.push_back(a1);
          kalmanGain3_.push_back(a2);
          kalmanGain3_.push_back(a3);
          kalmanGain3_.push_back(a4);
          kalmanGain3_.push_back(a5);
          kalmanGain3_.push_back(a6);
          break;
        case 2:
          kalmanGain2_.push_back(K);
          kalmanGain2_.push_back(a1);
          kalmanGain2_.push_back(a2);
          kalmanGain2_.push_back(a3);
          kalmanGain2_.push_back(a4);
          kalmanGain2_.push_back(a5);
          kalmanGain2_.push_back(a6);
          break;
        case 1:
          kalmanGain1_.push_back(K);
          kalmanGain1_.push_back(a1);
          kalmanGain1_.push_back(a2);
          kalmanGain1_.push_back(a3);
          kalmanGain1_.push_back(a4);
          kalmanGain1_.push_back(a5);
          kalmanGain1_.push_back(a6);
          break;
        case 0:
          kalmanGain0_.push_back(K);
          kalmanGain0_.push_back(a1);
          kalmanGain0_.push_back(a2);
          kalmanGain0_.push_back(a3);
          break;

        default:
          printf("Critical ERROR on setting the Kalman gain\n");
      }
    }

    //set covariance
    void setCovariance(const CovarianceMatrix& c) {
      covariance_[0] = c(0, 0);
      covariance_[1] = c(0, 1);
      covariance_[2] = c(1, 1);
      covariance_[3] = c(0, 2);
      covariance_[4] = c(1, 2);
      covariance_[5] = c(2, 2);
    }

    //set fine eta
    void setFineEta(int eta) {
      fineEta_ = eta;
      hasFineEta_ = true;
    }

    //set residual
    void setResidual(uint i, int val) { residuals_[i] = val; }

  private:
    reco::LeafCandidate::PolarLorentzVector unconstrainedP4_;

    //Covariance matrix for studies
    std::vector<double> covariance_;
    l1t::MuonStubRefVector stubs_;

    //vertex coordinates
    int curvVertex_;
    int ptC_;
    int phiVertex_;
    int dxy_;

    //muon coordinates
    int curvMuon_;
    int ptU_;
    int phiMuon_;
    int phiBMuon_;

    //generic coordinates
    int curv_;
    int phi_;
    int phiB_;
    //common coordinates
    int coarseEta_;

    //Approximate Chi2 metric
    int approxPromptChi2_;
    int approxPromptErrChi2_;
    int approxDispChi2_;
    int approxDispErrChi2_;

    //phi bitmask
    int hitPattern_;

    //propagation step
    int step_;

    //sector
    int sector_;
    //wheel
    int wheel_;

    //quality
    int quality_;

    //Fine eta
    int fineEta_;

    //has fine eta?
    bool hasFineEta_;

    //BX
    int bx_;

    //rank
    int rankPrompt_;
    int rankDisp_;

    //flag
    int idFlag_;

    //Kalman Gain for making LUTs
    std::vector<float> kalmanGain0_;
    std::vector<float> kalmanGain1_;
    std::vector<float> kalmanGain2_;
    std::vector<float> kalmanGain3_;

    std::vector<int> residuals_;
  };

}  // namespace l1t
#endif
