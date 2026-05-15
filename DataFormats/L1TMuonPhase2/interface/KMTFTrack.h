#ifndef DataFormats_L1TMuonPhase2_KMTFTrack_h
#define DataFormats_L1TMuonPhase2_KMTFTrack_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class KMTFTrack;
  typedef std::vector<KMTFTrack> KMTFTrackCollection;
  typedef BXVector<KMTFTrack> KMTFTrackBxCollection;
  typedef math::Error<5>::type CovarianceMatrix5dim;
  typedef math::Error<2>::type CovarianceMatrix2dim;

  class KMTFTrack : public reco::LeafCandidate {
  public:
    KMTFTrack()
        : reco::LeafCandidate(-1, reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          unconstrainedP4_(reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          covariance_(std::vector<double>(15, 0.0)),
		  covarianceNB_(std::vector<double>(3, 0.0)),
          curvVertex_(0),
          ptC_(0),
          phiVertex_(0),
          dxy_(0),
          zVertex_(0),
          kSlopeVertex_(0),
          curvMuon_(0),
          ptU_(0),
          phiMuon_(0),
          phiBMuon_(0),
          zMuon_(0),
          kSlopeMuon_(0),
          curv_(0),
          phi_(0),
          phiB_(0),
		  z_(0),
		  kSlope_(0),
          coarseEta_(0),
          approxPromptChi2_(0),
          approxPromptErrChi2_(0),
          approxDispChi2_(0),
          approxDispErrChi2_(0),
          hitPattern_(0),
          thetaDigiPattern_(0),
          step_(1),
          sector_(0),
          wheel_(0),
          quality_(0),
          hasFineEta_(false),
          bx_(0),
          rankPrompt_(0),
          rankDisp_(0),
          idFlag_(0) {}

    ~KMTFTrack() override = default;

    KMTFTrack(const l1t::MuonStubRef& seed, int phi, int phiB, int z, int kSlope)
        : reco::LeafCandidate(-1, reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          unconstrainedP4_(reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
          covariance_(std::vector<double>(15, 0.0)),
		  covarianceNB_(std::vector<double>(3, 0.0)),
          curvVertex_(0),
          ptC_(0),
          phiVertex_(0),
          dxy_(0),
          zVertex_(0),
          kSlopeVertex_(0),
          curvMuon_(0),
          ptU_(0),
          phiMuon_(0),
          phiBMuon_(0),
          zMuon_(0),
          kSlopeMuon_(0),
          curv_(0),
          phi_(phi),
          phiB_(phiB),
		  z_(z),
		  kSlope_(kSlope),
          coarseEta_(0),
          approxPromptChi2_(0),
          approxPromptErrChi2_(0),
          approxDispChi2_(0),
          approxDispErrChi2_(0),
          hitPattern_(0),
          thetaDigiPattern_(0),
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
    //unconstrained z at station 1
    int zAtMuon() const { return zMuon_; }
    //unconstrained kSLope at station 1
    int kSlopeAtMuon() const { return kSlopeMuon_; }

    //constrained pt
    int ptPrompt() const { return ptC_; }
    //Constrained curvature at vertex
    int curvatureAtVertex() const { return curvVertex_; }
    //constrained phi at the vertex
    int phiAtVertex() const { return phiVertex_; }
    //constrained z at the vertex
    int zAtVertex() const { return zVertex_; }
    //constrained kSlope at the vertex
    int kSlopeAtVertex() const { return kSlopeVertex_; }
    //Impact parameter as calculated from the muon track
    int dxy() const { return dxy_; }
    //Unconstrained curvature at the Muon systen
    int curvature() const { return curv_; }
    //Unconstrained phi at the Muon systen
    int positionAngle() const { return phi_; }
    //Unconstrained bending angle at the Muon systen
    int bendingAngle() const { return phiB_; }
	
	//global z and slope of stub 
	int zPosition() const {return z_;}
	int kSlope() const {return kSlope_;}
    //Coarse eta caluclated only using phi segments
    int coarseEta() const { return coarseEta_; }
    //Approximate Chi2 metrics
    int approxPromptChi2() const { return approxPromptChi2_; }
    int approxPromptErrChi2() const { return approxPromptErrChi2_; }
    int approxDispChi2() const { return approxDispChi2_; }
    int approxDispErrChi2() const { return approxDispErrChi2_; }

    int hitPattern() const { return hitPattern_; }
    int thetaDigiPattern() const { return thetaDigiPattern_; }
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

	const std::vector<float>& ThetaGain1D(unsigned int step) const {
		switch (step) {
    		case 3: return ThetaGain1D3_;
    		case 2: return ThetaGain1D2_;
    		case 1: return ThetaGain1D1_;
    		case 0: return ThetaGain1D0_;
		}
		return ThetaGain1D0_;
	}

    const std::vector<float>& ThetaGain(unsigned int step) const {
        switch (step) {
            case 3: return ThetaGain3_;
            case 2: return ThetaGain2_;
            case 1: return ThetaGain1_;
            case 0: return ThetaGain0_;
        }
        return ThetaGain0_;
    }


    //get covariance
    const std::vector<double>& covariance() const { return covariance_; }
	const std::vector<double>& covarianceNB() const { return covarianceNB_; }

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
    void setCoordinates(int step, int curv, int phi, int phiB, int z, int kSlope) {
      step_ = step;
      curv_ = curv;
      phiB_ = phiB;
      phi_ = phi;
	  z_ = z;
	  kSlope_ = kSlope;
    }

    void setCoordinatesAtVertex(int curv, int phi, int dxy, int z, int kSlope) {
      curvVertex_ = curv;
      phiVertex_ = phi;
      dxy_ = dxy;
	  zVertex_ = z;
      kSlopeVertex_ = kSlope;
    }

    void setCoordinatesAtMuon(int curv, int phi, int phiB, int z, int kSlope) {
      curvMuon_ = curv;
      phiMuon_ = phi;
      phiBMuon_ = phiB;
	  zMuon_ = z;
      kSlopeMuon_ = kSlope;
    }

    void setPt(int ptC, int ptU) {
      ptC_ = ptC;
      ptU_ = ptU;
    }

    void setCoarseEta(int eta) { coarseEta_ = eta; }

    void setHitPattern(int pattern) { hitPattern_ = pattern; }
	void setThetaDigiPattern(int theta_pattern) { thetaDigiPattern_ = theta_pattern; }

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
          throw cms::Exception("WrongCondition") << "Critical ERROR on setting the Kalman gain\n";
      }
    }

	void setThetaGain1D(unsigned int step, unsigned int K, int priorThetaPattern, int seedStation, int priorPhiPattern, float G31, float G32, float G41, float G42) {
      std::vector<float>* v1 = nullptr;
      switch (step) {
        case 3: v1 = &ThetaGain1D3_; break;
        case 2: v1 = &ThetaGain1D2_; break;
        case 1: v1 = &ThetaGain1D1_; break;
        case 0: v1 = &ThetaGain1D0_; break;
        default:
          throw cms::Exception("WrongCondition") << "1D: Critical ERROR on setting the Theta gain\n";
      }
      v1->push_back(static_cast<float>(K));
      v1->push_back(static_cast<float>(priorThetaPattern));
      v1->push_back(static_cast<float>(seedStation));
      v1->push_back(static_cast<float>(priorPhiPattern));
      v1->push_back(G31);
      v1->push_back(G32);
      v1->push_back(G41);
      v1->push_back(G42);
    }

    void setThetaGain(unsigned int step, unsigned int K, int priorThetaPattern, int seedStation, int priorPhiPattern, float G32, float G33, float G42, float G43) {
      std::vector<float>* v2 = nullptr;
      switch (step) {
        case 3: v2 = &ThetaGain3_; break;
        case 2: v2 = &ThetaGain2_; break;
        case 1: v2 = &ThetaGain1_; break;
        case 0: v2 = &ThetaGain0_; break;
        default:
          throw cms::Exception("WrongCondition") << "Critical ERROR on setting the Theta gain\n";
      }
      v2->push_back(static_cast<float>(K));
      v2->push_back(static_cast<float>(priorThetaPattern));
      v2->push_back(static_cast<float>(seedStation));
      v2->push_back(static_cast<float>(priorPhiPattern));
      v2->push_back(G32);
      v2->push_back(G33);
      v2->push_back(G42);
      v2->push_back(G43);
    }

    //set covariance
    void setCovariance(const CovarianceMatrix5dim& c) {
      covariance_[0] = c(0, 0);
      covariance_[1] = c(0, 1);
      covariance_[2] = c(1, 1);
      covariance_[3] = c(0, 2);
      covariance_[4] = c(1, 2);
      covariance_[5] = c(2, 2);
      covariance_[6] = c(0, 3);
      covariance_[7] = c(1, 3);
      covariance_[8] = c(2, 3);
      covariance_[9] = c(3, 3);
      covariance_[10] = c(0, 4);
      covariance_[11] = c(1, 4);
      covariance_[12] = c(2, 4);
      covariance_[13] = c(3, 4);
      covariance_[14] = c(4, 4);
	
    }
	
	void setCovarianceNB(const CovarianceMatrix2dim& c) {
  		covarianceNB_[0] = c(0,0);
  		covarianceNB_[1] = c(0,1);
  		covarianceNB_[2] = c(1,1);
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
	std::vector<double> covarianceNB_; 
    l1t::MuonStubRefVector stubs_;

    //vertex coordinates
    int curvVertex_;
    int ptC_;
    int phiVertex_;
    int dxy_;
    int zVertex_;
    int kSlopeVertex_;

    //muon coordinates
    int curvMuon_;
    int ptU_;
    int phiMuon_;
    int phiBMuon_;
    int zMuon_;
    int kSlopeMuon_;

    //generic coordinates
    int curv_;
    int phi_;
    int phiB_;
	int z_;
	int kSlope_;
    //common coordinates
    int coarseEta_;

    //Approximate Chi2 metric
    int approxPromptChi2_;
    int approxPromptErrChi2_;
    int approxDispChi2_;
    int approxDispErrChi2_;

    //phi bitmask
    int hitPattern_;
	//bitmask pattern based on theta digi presence
    int thetaDigiPattern_;

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

	std::vector<float> ThetaGain1D0_;
	std::vector<float> ThetaGain1D1_;
	std::vector<float> ThetaGain1D2_;
	std::vector<float> ThetaGain1D3_;

	std::vector<float> ThetaGain0_;
	std::vector<float> ThetaGain1_;
	std::vector<float> ThetaGain2_;
	std::vector<float> ThetaGain3_;
  };

}  // namespace l1t
#endif
