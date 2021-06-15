#ifndef __AnalysisDataFormats_PackedGenParticle_h__
#define __AnalysisDataFormats_PackedGenParticle_h__

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
/* #include "DataFormats/Math/interface/PtEtaPhiMass.h" */

class testPackedGenParticle;

namespace pat {
  class PackedGenParticle : public reco::Candidate {
  public:
    friend class ::testPackedGenParticle;

    /// collection of daughter candidates
    typedef reco::CandidateCollection daughters;
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// Lorentz vector
    typedef math::PtEtaPhiMLorentzVector PolarLorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// point in the space
    typedef math::XYZVector Vector;

    typedef unsigned int index;

    /// default constructor
    PackedGenParticle()
        : packedPt_(0),
          packedY_(0),
          packedPhi_(0),
          packedM_(0),
          p4_(nullptr),
          p4c_(nullptr),
          vertex_(0, 0, 0),
          pdgId_(0),
          charge_(0) {}
    explicit PackedGenParticle(const reco::GenParticle& c)
        : p4_(new PolarLorentzVector(c.pt(), c.eta(), c.phi(), c.mass())),
          p4c_(new LorentzVector(*p4_)),
          vertex_(0, 0, 0),
          pdgId_(c.pdgId()),
          charge_(c.charge()),
          mother_(c.motherRef(0)),
          statusFlags_(c.statusFlags()) {
      pack();
    }
    explicit PackedGenParticle(const reco::GenParticle& c, const edm::Ref<reco::GenParticleCollection>& mother)
        : p4_(new PolarLorentzVector(c.pt(), c.eta(), c.phi(), c.mass())),
          p4c_(new LorentzVector(*p4_)),
          vertex_(0, 0, 0),
          pdgId_(c.pdgId()),
          charge_(c.charge()),
          mother_(mother),
          statusFlags_(c.statusFlags()) {
      pack();
    }

    PackedGenParticle(const PackedGenParticle& iOther)
        : packedPt_(iOther.packedPt_),
          packedY_(iOther.packedY_),
          packedPhi_(iOther.packedPhi_),
          packedM_(iOther.packedM_),
          p4_(nullptr),
          p4c_(nullptr),
          vertex_(iOther.vertex_),
          dxy_(iOther.dxy_),
          dz_(iOther.dz_),
          dphi_(iOther.dphi_),
          pdgId_(iOther.pdgId_),
          charge_(iOther.charge_),
          mother_(iOther.mother_),
          statusFlags_(iOther.statusFlags_) {
      if (iOther.p4c_) {
        p4_.store(new PolarLorentzVector(*iOther.p4_));
        p4c_.store(new LorentzVector(*iOther.p4c_));
      }
    }

    PackedGenParticle(PackedGenParticle&& iOther)
        : packedPt_(iOther.packedPt_),
          packedY_(iOther.packedY_),
          packedPhi_(iOther.packedPhi_),
          packedM_(iOther.packedM_),
          p4_(nullptr),
          p4c_(nullptr),
          vertex_(std::move(iOther.vertex_)),
          dxy_(iOther.dxy_),
          dz_(iOther.dz_),
          dphi_(iOther.dphi_),
          pdgId_(iOther.pdgId_),
          charge_(iOther.charge_),
          mother_(iOther.mother_),
          statusFlags_(iOther.statusFlags_) {
      if (iOther.p4c_) {
        p4_.store(p4_.exchange(nullptr));
        p4c_.store(p4c_.exchange(nullptr));
      }
    }

    PackedGenParticle& operator=(PackedGenParticle&& iOther) {
      if (this != &iOther) {
        packedPt_ = iOther.packedPt_;
        packedY_ = iOther.packedY_;
        packedPhi_ = iOther.packedPhi_;
        packedM_ = iOther.packedM_;
        if (p4c_) {
          delete p4_.exchange(iOther.p4_.exchange(nullptr));
          delete p4c_.exchange(iOther.p4c_.exchange(nullptr));
        } else {
          delete p4_.exchange(nullptr);
          delete p4c_.exchange(nullptr);
        }
        vertex_ = std::move(iOther.vertex_);
        dxy_ = iOther.dxy_;
        dz_ = iOther.dz_;
        dphi_ = iOther.dphi_;
        pdgId_ = iOther.pdgId_;
        charge_ = iOther.charge_;
        mother_ = iOther.mother_;
        statusFlags_ = iOther.statusFlags_;
      }
      return *this;
    }

    PackedGenParticle& operator=(PackedGenParticle const& iOther) {
      PackedGenParticle c(iOther);
      *this = std::move(c);
      return *this;
    }

    /// destructor
    ~PackedGenParticle() override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// return daughter at a given position (throws an exception)
    const reco::Candidate* daughter(size_type) const override;
    /// number of mothers
    size_t numberOfMothers() const override;
    /// return mother at a given position (throws an exception)
    const reco::Candidate* mother(size_type) const override;
    /// direct access to the mother reference (may be null)
    reco::GenParticleRef motherRef() const {
      if (mother_.isNonnull() && mother_.isAvailable() &&
          mother_->status() == 1) {  //if pointing to the pruned version of myself
        if (mother_->numberOfMothers() > 0)
          return mother_->motherRef(0);  // return my mother's (that is actually myself) mother
        else
          return edm::Ref<reco::GenParticleCollection>();  // return null ref
      } else {
        return mother_;  //the stored ref is really my mother, or null, return that
      }
    }
    /// last surviving in pruned
    const reco::GenParticleRef& lastPrunedRef() const { return mother_; }

    /// return daughter at a given position (throws an exception)
    reco::Candidate* daughter(size_type) override;
    /// return daughter with a specified role name
    reco::Candidate* daughter(const std::string& s) override;
    /// return daughter with a specified role name
    const reco::Candidate* daughter(const std::string& s) const override;
    /// return the number of source Candidates
    /// ( the candidates used to construct this Candidate)
    size_t numberOfSourceCandidatePtrs() const override { return 0; }
    /// return a Ptr to one of the source Candidates
    /// ( the candidates used to construct this Candidate)
    reco::CandidatePtr sourceCandidatePtr(size_type i) const override { return reco::CandidatePtr(); }

    /// electric charge
    int charge() const override { return charge_; }
    /// set electric charge
    void setCharge(int charge) override { charge_ = charge; }
    /// electric charge
    int threeCharge() const override { return charge() * 3; }
    /// set electric charge
    void setThreeCharge(int threecharge) override {}
    /// four-momentum Lorentz vecto r
    const LorentzVector& p4() const override {
      if (!p4c_)
        unpack();
      return *p4c_;
    }
    /// four-momentum Lorentz vector
    const PolarLorentzVector& polarP4() const override {
      if (!p4c_)
        unpack();
      return *p4_;
    }
    /// spatial momentum vector
    Vector momentum() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->Vect();
    }
    /// boost vector to boost a Lorentz vector
    /// to the particle center of mass system
    Vector boostToCM() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->BoostToCM();
    }
    /// magnitude of momentum vector
    double p() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->P();
    }
    /// energy
    double energy() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->E();
    }
    /// transverse energy
    double et() const override { return (pt() <= 0) ? 0 : p4c_.load()->Et(); }
    /// transverse energy squared (use this for cuts)!
    double et2() const override { return (pt() <= 0) ? 0 : p4c_.load()->Et2(); }
    /// mass
    double mass() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->M();
    }
    /// mass squared
    double massSqr() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->M() * p4_.load()->M();
    }

    /// transverse mass
    double mt() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Mt();
    }
    /// transverse mass squared
    double mtSqr() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Mt2();
    }
    /// x coordinate of momentum vector
    double px() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->Px();
    }
    /// y coordinate of momentum vector
    double py() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->Py();
    }
    /// z coordinate of momentum vector
    double pz() const override {
      if (!p4c_)
        unpack();
      return p4c_.load()->Pz();
    }
    /// transverse momentum
    double pt() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Pt();
    }
    /// momentum azimuthal angle
    double phi() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Phi();
    }
    /// momentum polar angle
    double theta() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Theta();
    }
    /// momentum pseudorapidity
    double eta() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Eta();
    }
    /// rapidity
    double rapidity() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Rapidity();
    }
    /// rapidity
    double y() const override {
      if (!p4c_)
        unpack();
      return p4_.load()->Rapidity();
    }
    /// set 4-momentum
    void setP4(const LorentzVector& p4) override {
      unpack();  // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      *p4_ = PolarLorentzVector(p4.Pt(), p4.Eta(), p4.Phi(), p4.M());
      pack();
    }
    /// set 4-momentum
    void setP4(const PolarLorentzVector& p4) override {
      unpack();  // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      *p4_ = p4;
      pack();
    }
    /// set particle mass
    void setMass(double m) override {
      if (!p4c_)
        unpack();
      *p4_ = PolarLorentzVector(p4_.load()->Pt(), p4_.load()->Eta(), p4_.load()->Phi(), m);
      pack();
    }
    void setPz(double pz) override {
      unpack();  // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      *p4c_ = LorentzVector(p4c_.load()->Px(), p4c_.load()->Py(), pz, p4c_.load()->E());
      *p4_ = PolarLorentzVector(p4c_.load()->Pt(), p4c_.load()->Eta(), p4c_.load()->Phi(), p4c_.load()->M());
      pack();
    }
    /// vertex position
    const Point& vertex() const override {
      return vertex_;
    }  //{ if (fromPV_) return Point(0,0,0); else return Point(0,0,100); }
    /// x coordinate of vertex position
    double vx() const override { return vertex_.X(); }  //{ return 0; }
    /// y coordinate of vertex position
    double vy() const override { return vertex_.Y(); }  //{ return 0; }
    /// z coordinate of vertex position
    double vz() const override { return vertex_.Z(); }  //{ if (fromPV_) return 0; else return 100; }
    /// set vertex
    void setVertex(const Point& vertex) override { vertex_ = vertex; }

    enum PVAssoc { NoPV = 0, PVLoose = 1, PVTight = 2, PVUsedInFit = 3 };

    /// dxy with respect to the PV ref
    virtual float dxy() const {
      unpack();
      return dxy_;
    }
    /// dz with respect to the PV ref
    virtual float dz() const {
      unpack();
      return dz_;
    }
    /// dxy with respect to another point
    virtual float dxy(const Point& p) const;
    /// dz  with respect to another point
    virtual float dz(const Point& p) const;

    /// PDG identifier
    int pdgId() const override { return pdgId_; }
    // set PDG identifier
    void setPdgId(int pdgId) override { pdgId_ = pdgId; }
    /// status word
    int status() const override { return 1; } /*FIXME*/
    /// set status word
    void setStatus(int status) override {} /*FIXME*/
    /// long lived flag
    static const unsigned int longLivedTag = 0; /*FIXME*/
    /// set long lived flag
    void setLongLived() override {} /*FIXME*/
    /// is long lived?
    bool longLived() const override;
    /// do mass constraint flag
    static const unsigned int massConstraintTag = 0; /*FIXME*/
    /// set mass constraint flag
    void setMassConstraint() override {} /*FIXME*/
    /// do mass constraint?
    bool massConstraint() const override;

    /// returns a clone of the Candidate object
    PackedGenParticle* clone() const override { return new PackedGenParticle(*this); }

    /// chi-squares
    double vertexChi2() const override;
    /** Number of degrees of freedom                                                                                   
     *  Meant to be Double32_t for soft-assignment fitters:                                                            
     *  tracks may contribute to the vertex with fractional weights.                                                   
     *  The ndof is then = to the sum of the track weights.                                                            
     *  see e.g. CMS NOTE-2006/032, CMS NOTE-2004/002                                                                  
     */
    double vertexNdof() const override;
    /// chi-squared divided by n.d.o.f.
    double vertexNormalizedChi2() const override;
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    double vertexCovariance(int i, int j) const override;
    /// return SMatrix
    CovarianceMatrix vertexCovariance() const override {
      CovarianceMatrix m;
      fillVertexCovariance(m);
      return m;
    }
    /// fill SMatrix
    void fillVertexCovariance(CovarianceMatrix& v) const override;
    /// returns true if this candidate has a reference to a master clone.
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate
    bool hasMasterClone() const override;
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate
    const reco::CandidateBaseRef& masterClone() const override;
    /// returns true if this candidate has a ptr to a master clone.
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate
    bool hasMasterClonePtr() const override;
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate

    const reco::CandidatePtr& masterClonePtr() const override;

    /// cast master clone reference to a concrete type
    template <typename Ref>
    Ref masterRef() const {
      return masterClone().template castTo<Ref>();
    }
    /// get a component

    bool isElectron() const override;
    bool isMuon() const override;
    bool isStandAloneMuon() const override;
    bool isGlobalMuon() const override;
    bool isTrackerMuon() const override;
    bool isCaloMuon() const override;
    bool isPhoton() const override;
    bool isConvertedPhoton() const override;
    bool isJet() const override;

    const reco::GenStatusFlags& statusFlags() const { return statusFlags_; }
    reco::GenStatusFlags& statusFlags() { return statusFlags_; }

    /////////////////////////////////////////////////////////////////////////////
    //basic set of gen status flags accessible directly here
    //the rest accessible through statusFlags()
    //(see GenStatusFlags.h for their meaning)

    /////////////////////////////////////////////////////////////////////////////
    //these are robust, generator-independent functions for categorizing
    //mainly final state particles, but also intermediate hadrons/taus

    //is particle prompt (not from hadron, muon, or tau decay) and final state
    bool isPromptFinalState() const { return status() == 1 && statusFlags_.isPrompt(); }

    //this particle is a direct decay product of a prompt tau and is final state
    //(eg an electron or muon from a leptonic decay of a prompt tau)
    bool isDirectPromptTauDecayProductFinalState() const {
      return status() == 1 && statusFlags_.isDirectPromptTauDecayProduct();
    }

    /////////////////////////////////////////////////////////////////////////////
    //these are generator history-dependent functions for tagging particles
    //associated with the hard process
    //Currently implemented for Pythia 6 and Pythia 8 status codes and history
    //and may not have 100% consistent meaning across all types of processes
    //Users are strongly encouraged to stick to the more robust flags above,
    //as well as the expanded set available in GenStatusFlags.h

    //this particle is the final state direct descendant of a hard process particle
    bool fromHardProcessFinalState() const { return status() == 1 && statusFlags_.fromHardProcess(); }

    //this particle is a direct decay product of a hardprocess tau and is final state
    //(eg an electron or muon from a leptonic decay of a tau from the hard process)
    bool isDirectHardProcessTauDecayProductFinalState() const {
      return status() == 1 && statusFlags_.isDirectHardProcessTauDecayProduct();
    }

  protected:
    uint16_t packedPt_, packedY_, packedPhi_, packedM_;
    void pack(bool unpackAfterwards = true);
    void unpack() const;

    /// the four vector
    mutable std::atomic<PolarLorentzVector*> p4_;
    mutable std::atomic<LorentzVector*> p4c_;
    /// vertex position
    Point vertex_;
    float dxy_, dz_, dphi_;
    /// PDG identifier
    int pdgId_;
    /// Charge
    int8_t charge_;
    ///Ref to first mother
    reco::GenParticleRef mother_;
    //status flags
    reco::GenStatusFlags statusFlags_;

    /// check overlap with another Candidate
    bool overlap(const reco::Candidate&) const override;
    template <typename, typename, typename>
    friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;
  };

  typedef std::vector<pat::PackedGenParticle> PackedGenParticleCollection;
  typedef edm::Ref<pat::PackedGenParticleCollection> PackedGenParticleRef;
  typedef edm::RefVector<pat::PackedGenParticleCollection> PackedGenParticleRefVector;
}  // namespace pat

#endif
