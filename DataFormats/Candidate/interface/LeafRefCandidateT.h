#ifndef Candidate_LeafRefCandidateT_h
#define Candidate_LeafRefCandidateT_h
/** \class reco::LeafRefCandidateT
 *
 * particle candidate with no constituent nor daughters, that takes the 3-vector
 * from a constituent T (where T satisfies T->pt(), etc, like a TrackRef), and the mass is set
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: LeafRefCandidateT.h,v 1.2 2011/10/27 19:54:25 wmtan Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

#include "DataFormats/Common/interface/BoolCache.h"

namespace reco {
  
  template < class T >
  class LeafRefCandidateT : public Candidate {
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
    LeafRefCandidateT() : 
      mass_(0), 
      cachePolarFixed_( false ) { }
    // constructor from T                                                        
    explicit LeafRefCandidateT( const T & c, float m) :
      ref_(c),
      mass_( m ),
      cachePolarFixed_( false ), cacheCartesianFixed_( false ) {}

    /// destructor
    virtual ~LeafRefCandidateT() {}
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter iterator
    virtual iterator end();
    /// number of daughters
    virtual size_t numberOfDaughters() const { return 0; }
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * daughter( size_type ) const { return 0; }
    /// number of mothers
    virtual size_t numberOfMothers() const { return 0; }
    /// return mother at a given position (throws an exception)
    virtual const Candidate * mother( size_type ) const { return 0; }
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type ) { return 0; }
    /// return daughter with a specified role name
    virtual Candidate * daughter(const std::string& s ) { return 0; }
    /// return daughter with a specified role name                                        
    virtual const Candidate * daughter(const std::string& s ) const { return 0; }
    /// return the number of source Candidates                                            
    /// ( the candidates used to construct this Candidate)                                
    virtual size_t numberOfSourceCandidatePtrs() const { return 0;}
    /// return a Ptr to one of the source Candidates                                      
    /// ( the candidates used to construct this Candidate)                                
    virtual CandidatePtr sourceCandidatePtr( size_type i ) const {
      static CandidatePtr dummyPtr;
      return dummyPtr;
    }

    /// electric charge
    virtual int charge() const { return ref_->charge(); }
    /// pdg ID: dummy for now
    virtual int pdgId() const { return 0; }

    /// four-momentum Lorentz vector                                                      
    virtual const LorentzVector & p4() const { cacheCartesian(); return p4Cartesian_; }
    /// four-momentum Lorentz vector                                                      
    virtual const PolarLorentzVector & polarP4() const { cachePolar(); return p4Polar_; }
    /// spatial momentum vector                                                           
    virtual Vector momentum() const { cacheCartesian(); return p4Cartesian_.Vect(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    virtual Vector boostToCM() const { cacheCartesian(); return p4Cartesian_.BoostToCM(); }
    /// magnitude of momentum vector                                                      
    virtual double p() const { cacheCartesian(); return p4Cartesian_.P(); }
    /// energy                                                                            
    virtual double energy() const { cacheCartesian(); return p4Cartesian_.E(); }
    /// transverse energy                                                                 
    virtual double et() const { cachePolar(); return p4Polar_.Et(); }
    /// mass                                                                              
    virtual double mass() const { return mass_; }
    /// mass squared                                                                      
    virtual double massSqr() const { return mass_ * mass_; }
    /// transverse mass                                                                   
    virtual double mt() const { cachePolar(); return p4Polar_.Mt(); }
    /// transverse mass squared                                                           
    virtual double mtSqr() const { cachePolar(); return p4Polar_.Mt2(); }
    /// x coordinate of momentum vector                                                   
    virtual double px() const { cacheCartesian(); return p4Cartesian_.Px(); }
    /// y coordinate of momentum vector                                                   
    virtual double py() const { cacheCartesian(); return p4Cartesian_.Py(); }
    /// z coordinate of momentum vector                                                   
    virtual double pz() const { cacheCartesian(); return p4Cartesian_.Pz(); }
    /// transverse momentum                                                               
    virtual double pt() const { return ref_->pt(); } 
    /// momentum azimuthal angle                                                          
    virtual double phi() const { return ref_->phi(); }
    /// momentum polar angle                                                              
    virtual double theta() const { return ref_->theta(); }
    /// momentum pseudorapidity                                                           
    virtual double eta() const { return ref_->eta(); }
    /// rapidity                                                                          
    virtual double rapidity() const { cachePolar(); return p4Polar_.Rapidity(); }
    /// rapidity                                                                          
    virtual double y() const { return rapidity(); }

    /// set particle mass                                                                 
    virtual void setMass( double m ) {
      mass_ = m;
      clearCache();
    }

    /// vertex position                                                                   
    virtual const Point & vertex() const { return ref_->vertex(); }
    /// x coordinate of vertex position                                                   
    virtual double vx() const { return ref_->vx(); }
    /// y coordinate of vertex position                                                   
    virtual double vy() const { return ref_->vy(); }
    /// z coordinate of vertex position                                                   
    virtual double vz() const { return ref_->vz(); }


    /// returns a clone of the Candidate object                                           
    virtual LeafRefCandidateT<T> * clone() const {
      return new LeafRefCandidateT<T>( *this );
    }

                     
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate                                      
    virtual bool hasMasterClone() const { return false; }
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate                                
    virtual const CandidateBaseRef & masterClone() const { 
      static CandidateBaseRef dummyRef; return dummyRef; 
    }
    /// returns true if this candidate has a ptr to a master clone.                                                    
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate                                   
    virtual bool hasMasterClonePtr() const { return false; }
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate                             
    virtual const CandidatePtr & masterClonePtr() const { 
      static CandidatePtr dummyPtr; return dummyPtr; 
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

    template<typename S>
      struct daughter_iterator {
        typedef boost::filter_iterator<S, const_iterator> type;
      };

    template<typename S>
      typename daughter_iterator<S>::type beginFilter( const S & s ) const {
      return boost::make_filter_iterator(s, begin(), end());
    }
    template<typename S>
      typename daughter_iterator<S>::type endFilter( const S & s ) const {
      return boost::make_filter_iterator(s, end(), end());
    }


    virtual bool isElectron() const { return false; }
    virtual bool isMuon() const { return false; }
    virtual bool isStandAloneMuon() const { return false; }
    virtual bool isGlobalMuon() const { return false; }
    virtual bool isTrackerMuon() const { return false; }
    virtual bool isCaloMuon() const { return false; }
    virtual bool isPhoton() const { return false; }
    virtual bool isConvertedPhoton() const { return false; }
    virtual bool isJet() const { return false; }

  protected:
    /// T internally.
    /// NOTE! T must satisfy ref_->pt(), ref_->phi(), etc, like a TrackRef
    T    ref_;
    /// mass hypothesis                                                  
    float mass_;
    /// internal cache for p4                                                             
    mutable PolarLorentzVector p4Polar_;
    /// internal cache for p4                                                             
    mutable LorentzVector p4Cartesian_;
    /// has cache been set?                                                               
    mutable  edm::BoolCache cachePolarFixed_, cacheCartesianFixed_;
    /// set internal cache                                                                
    inline void cachePolar() const {
      if ( cachePolarFixed_ ) return;
      p4Polar_ = PolarLorentzVector( ref_->pt(), ref_->eta(), ref_->phi(), mass_ );
      cachePolarFixed_ = true;
    }
    /// set internal cache                                                                
    inline void cacheCartesian() const {
      if ( cacheCartesianFixed_ ) return;
      cachePolar();
      p4Cartesian_ = p4Polar_;
      cacheCartesianFixed_ = true;
    }
    /// clear internal cache                                                              
    inline void clearCache() const {
      cachePolarFixed_ = false;
      cacheCartesianFixed_ = false;
    }
    /// check overlap with another Candidate                                              
    virtual bool overlap( const Candidate & ) const;
    virtual bool overlap( const LeafRefCandidateT & ) const;
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  private:
    // const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    // iterator implementation
    typedef candidate::iterator_imp_specific<daughters> iterator_imp_specific;


    ///
    /// Hide these from all users:
    ///
                                        
    virtual void setCharge( Charge q ) { return; }                                         
    virtual int threeCharge() const { return 0; }
    virtual void setThreeCharge( Charge qx3 ) { return; }
    virtual void setP4( const LorentzVector & p4 ) {
      return;
    }
    virtual void setP4( const PolarLorentzVector & p4 ) {
      return;
    }
    virtual void setPz( double pz ) {
      return;
    }                            
    virtual void setVertex( const Point & vertex ) { return; }
    virtual void setPdgId( int pdgId ) { return; }                                              
    virtual int status() const { return 0; }                                      
    virtual void setStatus( int status ) { return; }
    static const unsigned int longLivedTag;                                               
    virtual void setLongLived() { return; }                                      
    virtual bool longLived() const { return false; }
    static const unsigned int massConstraintTag;
    virtual void setMassConstraint() { return;}
    virtual bool massConstraint() const { return false; }  
    virtual double vertexChi2() const { return 0.; }
    virtual double vertexNdof() const { return 0.; }
    virtual double vertexNormalizedChi2() const { return 0.; }
    virtual double vertexCovariance(int i, int j) const { return 0.; }
    CovarianceMatrix vertexCovariance() const { CovarianceMatrix m; return m; }
    virtual void fillVertexCovariance(CovarianceMatrix & v) const { return ; }
    
  };



  template<class T>
  Candidate::const_iterator LeafRefCandidateT<T>::begin() const { 
    return const_iterator( new const_iterator_imp_specific ); 
  }

  template<class T>
  Candidate::const_iterator LeafRefCandidateT<T>::end() const { 
    return  const_iterator( new const_iterator_imp_specific ); 
  }

  template<class T>
  Candidate::iterator LeafRefCandidateT<T>::begin() { 
    return iterator( new iterator_imp_specific ); 
  }
  
  template<class T>
  Candidate::iterator LeafRefCandidateT<T>::end() { 
    return iterator( new iterator_imp_specific ); 
  }
  
  
  template<class T>
  bool LeafRefCandidateT<T>::overlap( const Candidate & o ) const { 
    return  p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
  }
  
  
  template<class T>
  bool LeafRefCandidateT<T>::overlap( const LeafRefCandidateT & o ) const { 
    return  ref_ == o.ref_;
  }



}

#endif
