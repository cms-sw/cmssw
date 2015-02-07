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
    LeafRefCandidateT() { }
    // constructor from T                                                        
    template < class REF >
    LeafRefCandidateT( const REF & c, float m) :  
      LeafCandidate(c->charge(),PolarLorentzVector(c->pt(), c->eta(), c->phi(), m ),c->vertex()),
      ref_(c.refCore(), c.key()){}
    /// destructor
    virtual ~LeafRefCandidateT() {}

protected:
    // get the ref (better be the correct ref!)
    template<typename REF>   
    REF getRef() const { return REF(ref_.toRefCore(),ref_.index()); }

public:
    /// number of daughters
    virtual size_t numberOfDaughters() const GCC11_FINAL  { return 0; }
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * daughter( size_type ) const GCC11_FINAL  { return 0; }
    /// number of mothers
    virtual size_t numberOfMothers() const GCC11_FINAL  { return 0; }
    /// return mother at a given position (throws an exception)
    virtual const Candidate * mother( size_type ) const GCC11_FINAL  { return 0; }
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type ) GCC11_FINAL  { return 0; }
    /// return daughter with a specified role name
    virtual Candidate * daughter(const std::string& s ) GCC11_FINAL  { return 0; }
    /// return daughter with a specified role name                                        
    virtual const Candidate * daughter(const std::string& s ) const GCC11_FINAL  { return 0; }
    /// return the number of source Candidates                                            
    /// ( the candidates used to construct this Candidate)                                
    virtual size_t numberOfSourceCandidatePtrs() const GCC11_FINAL  { return 0;}
    /// return a Ptr to one of the source Candidates                                      
    /// ( the candidates used to construct this Candidate)                                
    virtual CandidatePtr sourceCandidatePtr( size_type i ) const GCC11_FINAL  {
      static const CandidatePtr dummyPtr;
      return dummyPtr;
    }


                     
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate                                      
    virtual bool hasMasterClone() const GCC11_FINAL  { return false; }
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate                                
    virtual const CandidateBaseRef & masterClone() const GCC11_FINAL  { 
      static const CandidateBaseRef dummyRef; return dummyRef; 
    }
    /// returns true if this candidate has a ptr to a master clone.                                                    
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate                                   
    virtual bool hasMasterClonePtr() const GCC11_FINAL  { return false; }
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate                             
    virtual const CandidatePtr & masterClonePtr() const GCC11_FINAL  { 
      static const CandidatePtr dummyPtr; return dummyPtr; 
    }

    /// cast master clone reference to a concrete type                                                                 
    template<typename Ref>
      Ref masterRef() const { Ref dummyRef; return dummyRef; }
    /// get a component                                                                                                

    template<typename C> C get() const {
      if ( hasMasterClone() ) return masterClone()->template get<C>();
      else return reco::get<C>( * this );
    }
    /// get a component                                                                                                
    template<typename C, typename Tag> C get() const {
      if ( hasMasterClone() ) return masterClone()->template get<C, Tag>();
      else return reco::get<C, Tag>( * this );
    }
    /// get a component                                                                                                
    template<typename C> C get( size_type i ) const {
      if ( hasMasterClone() ) return masterClone()->template get<C>( i );
      else return reco::get<C>( * this, i );
    }
    /// get a component                                                                                                
    template<typename C, typename Tag> C get( size_type i ) const {
      if ( hasMasterClone() ) return masterClone()->template get<C, Tag>( i );
      else return reco::get<C, Tag>( * this, i );
    }
    /// number of components                                                                                           
    template<typename C> size_type numberOf() const {
      if ( hasMasterClone() ) return masterClone()->template numberOf<C>();
      else return reco::numberOf<C>( * this );
    }
    /// number of components                                                                                           
    template<typename C, typename Tag> size_type numberOf() const {
      if ( hasMasterClone() ) return masterClone()->template numberOf<C, Tag>();
      else return reco::numberOf<C, Tag>( * this );
    }


    virtual bool isElectron() const GCC11_FINAL  { return false; }
    virtual bool isMuon() const GCC11_FINAL  { return false; }
    virtual bool isStandAloneMuon() const GCC11_FINAL  { return false; }
    virtual bool isGlobalMuon() const GCC11_FINAL  { return false; }
    virtual bool isTrackerMuon() const GCC11_FINAL  { return false; }
    virtual bool isCaloMuon() const GCC11_FINAL  { return false; }
    virtual bool isPhoton() const GCC11_FINAL  { return false; }
    virtual bool isConvertedPhoton() const GCC11_FINAL  { return false; }
    virtual bool isJet() const GCC11_FINAL  { return false; }

    CMS_CLASS_VERSION(13)

  protected:
 
    /// check overlap with another Candidate                                              
    virtual bool overlap( const Candidate & ) const;
    virtual bool overlap( const LeafRefCandidateT & ) const;
    template<typename, typename, typename> friend struct component;
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
    virtual void setCharge( Charge q ) GCC11_FINAL  {}                                         
    virtual void setThreeCharge( Charge qx3 ) GCC11_FINAL  {}
    virtual void setP4( const LorentzVector & p4 ) GCC11_FINAL  {}
    virtual void setP4( const PolarLorentzVector & p4 ) GCC11_FINAL  {}
    virtual void setPz( double pz ) GCC11_FINAL  {}                            
    virtual void setVertex( const Point & vertex ) GCC11_FINAL  {}
    virtual void setPdgId( int pdgId ) GCC11_FINAL  {}                                              
    virtual void setStatus( int status ) GCC11_FINAL  {}
    virtual void setLongLived() GCC11_FINAL  {}                                      
    virtual void setMassConstraint() GCC11_FINAL  {}

    virtual double vertexChi2() const GCC11_FINAL  { return 0.; }
    virtual double vertexNdof() const GCC11_FINAL  { return 0.; }
    virtual double vertexNormalizedChi2() const GCC11_FINAL  { return 0.; }
    virtual double vertexCovariance(int i, int j) const GCC11_FINAL  { return 0.; }
    virtual void fillVertexCovariance(CovarianceMatrix & v) const GCC11_FINAL  {}
    */
  };



  inline
  bool LeafRefCandidateT::overlap( const Candidate & o ) const  { 
    return  p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
  }
  
  inline
  bool LeafRefCandidateT::overlap( const LeafRefCandidateT & o ) const   { 
    return  (ref_.id() == o.ref_.id()) & (ref_.index() == o.ref_.index());
  }



}

#endif
