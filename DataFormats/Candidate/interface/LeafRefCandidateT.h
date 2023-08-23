#ifndef Candidate_LeafRefCandidateT_h
#define Candidate_LeafRefCandidateT_h
/** \class reco::LeafRefCandidateT
 *
 * particle candidate with no constituent nor daughters, that takes the 3-vector
 * from a constituent T (where T satisfies T->pt(), etc, like a TrackRef), and the mass is set
 *
 * \author Luca Lista, INFN
 *
 */

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/RefCoreWithIndex.h"

namespace reco {

  class LeafRefCandidateT : public LeafCandidate {
  public:
    /// collection of daughter candidates
    typedef CandidateCollection daughters;
    /// electric charge type
    typedef int Charge;
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
    LeafRefCandidateT() {}
    // constructor from T
    template <class REF>
    LeafRefCandidateT(const REF& c, float m)
        : LeafCandidate(c->charge(), PolarLorentzVector(c->pt(), c->eta(), c->phi(), m), c->vertex()),
          ref_(c.refCore(), c.key()) {}
    /// destructor
    ~LeafRefCandidateT() override {}

  protected:
    // get the ref (better be the correct ref!)
    template <typename REF>
    REF getRef() const {
      return REF(ref_.toRefCore(), ref_.index());
    }

  public:
    /// number of daughters
    size_t numberOfDaughters() const final { return 0; }
    /// return daughter at a given position (throws an exception)
    const Candidate* daughter(size_type) const final { return nullptr; }
    /// number of mothers
    size_t numberOfMothers() const final { return 0; }
    /// return mother at a given position (throws an exception)
    const Candidate* mother(size_type) const final { return nullptr; }
    /// return daughter at a given position (throws an exception)
    Candidate* daughter(size_type) final { return nullptr; }
    /// return daughter with a specified role name
    Candidate* daughter(const std::string& s) final { return nullptr; }
    /// return daughter with a specified role name
    const Candidate* daughter(const std::string& s) const final { return nullptr; }
    /// return the number of source Candidates
    /// ( the candidates used to construct this Candidate)
    size_t numberOfSourceCandidatePtrs() const final { return 0; }
    /// return a Ptr to one of the source Candidates
    /// ( the candidates used to construct this Candidate)
    CandidatePtr sourceCandidatePtr(size_type i) const final {
      static const CandidatePtr dummyPtr;
      return dummyPtr;
    }

    /// This only happens if the concrete Candidate type is ShallowCloneCandidate
    bool hasMasterClone() const final { return false; }
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate
    const CandidateBaseRef& masterClone() const final {
      static const CandidateBaseRef dummyRef;
      return dummyRef;
    }
    /// returns true if this candidate has a ptr to a master clone.
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate
    bool hasMasterClonePtr() const final { return false; }
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate
    const CandidatePtr& masterClonePtr() const final {
      static const CandidatePtr dummyPtr;
      return dummyPtr;
    }

    /// cast master clone reference to a concrete type
    template <typename Ref>
    Ref masterRef() const {
      Ref dummyRef;
      return dummyRef;
    }
    /// get a component

    template <typename C>
    C get() const {
      if (hasMasterClone())
        return masterClone()->template get<C>();
      else
        return reco::get<C>(*this);
    }
    /// get a component
    template <typename C, typename Tag>
    C get() const {
      if (hasMasterClone())
        return masterClone()->template get<C, Tag>();
      else
        return reco::get<C, Tag>(*this);
    }
    /// get a component
    template <typename C>
    C get(size_type i) const {
      if (hasMasterClone())
        return masterClone()->template get<C>(i);
      else
        return reco::get<C>(*this, i);
    }
    /// get a component
    template <typename C, typename Tag>
    C get(size_type i) const {
      if (hasMasterClone())
        return masterClone()->template get<C, Tag>(i);
      else
        return reco::get<C, Tag>(*this, i);
    }
    /// number of components
    template <typename C>
    size_type numberOf() const {
      if (hasMasterClone())
        return masterClone()->template numberOf<C>();
      else
        return reco::numberOf<C>(*this);
    }
    /// number of components
    template <typename C, typename Tag>
    size_type numberOf() const {
      if (hasMasterClone())
        return masterClone()->template numberOf<C, Tag>();
      else
        return reco::numberOf<C, Tag>(*this);
    }

    bool isElectron() const final { return false; }
    bool isMuon() const final { return false; }
    bool isStandAloneMuon() const final { return false; }
    bool isGlobalMuon() const final { return false; }
    bool isTrackerMuon() const final { return false; }
    bool isCaloMuon() const final { return false; }
    bool isPhoton() const final { return false; }
    bool isConvertedPhoton() const final { return false; }
    bool isJet() const final { return false; }

    CMS_CLASS_VERSION(13)

  protected:
    /// check overlap with another Candidate
    bool overlap(const Candidate&) const override;
    virtual bool overlap(const LeafRefCandidateT&) const;
    template <typename, typename, typename>
    friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  protected:
    edm::RefCoreWithIndex ref_;

  private:
    ///
    /// Hide these from all users:
    ///
    /*                                    
    virtual void setCharge( Charge q ) final  {}                                         
    virtual void setThreeCharge( Charge qx3 ) final  {}
    virtual void setP4( const LorentzVector & p4 ) final  {}
    virtual void setP4( const PolarLorentzVector & p4 ) final  {}
    virtual void setPz( double pz ) final  {}                            
    virtual void setVertex( const Point & vertex ) final  {}
    virtual void setPdgId( int pdgId ) final  {}                                              
    virtual void setStatus( int status ) final  {}
    virtual void setLongLived() final  {}                                      
    virtual void setMassConstraint() final  {}

    virtual double vertexChi2() const final  { return 0.; }
    virtual double vertexNdof() const final  { return 0.; }
    virtual double vertexNormalizedChi2() const final  { return 0.; }
    virtual double vertexCovariance(int i, int j) const final  { return 0.; }
    virtual void fillVertexCovariance(CovarianceMatrix & v) const final  {}
    */
  };

  inline bool LeafRefCandidateT::overlap(const Candidate& o) const {
    return (p4() == o.p4()) && (vertex() == o.vertex()) && (charge() == o.charge());
  }

  inline bool LeafRefCandidateT::overlap(const LeafRefCandidateT& o) const {
    return (ref_.id() == o.ref_.id()) && (ref_.index() == o.ref_.index());
  }

}  // namespace reco

#endif
