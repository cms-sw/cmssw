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
      static CandidatePtr dummyPtr;
      return dummyPtr;
    }

    /// electric charge
    virtual int charge() const GCC11_FINAL  { return ref_->charge(); }
    /// pdg ID: dummy for now
    virtual int pdgId() const GCC11_FINAL  { return 0; }

    /// four-momentum Lorentz vector                                                      
    virtual const LorentzVector & p4() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_; }
    /// four-momentum Lorentz vector                                                      
    virtual const PolarLorentzVector & polarP4() const GCC11_FINAL  { cachePolar(); return p4Polar_; }
    /// spatial momentum vector                                                           
    virtual Vector momentum() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.Vect(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    virtual Vector boostToCM() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.BoostToCM(); }
    /// magnitude of momentum vector                                                      
    virtual double p() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.P(); }
    /// energy                                                                            
    virtual double energy() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.E(); }
    /// transverse energy                                                                 
    virtual double et() const GCC11_FINAL  { cachePolar(); return p4Polar_.Et(); }
    /// mass                                                                              
    virtual float mass() const GCC11_FINAL  { return mass_; }
    /// mass squared                                                                      
    virtual float massSqr() const GCC11_FINAL  { return mass_ * mass_; }
    /// transverse mass                                                                   
    virtual double mt() const GCC11_FINAL  { cachePolar(); return p4Polar_.Mt(); }
    /// transverse mass squared                                                           
    virtual double mtSqr() const GCC11_FINAL  { cachePolar(); return p4Polar_.Mt2(); }
    /// x coordinate of momentum vector                                                   
    virtual double px() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.Px(); }
    /// y coordinate of momentum vector                                                   
    virtual double py() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.Py(); }
    /// z coordinate of momentum vector                                                   
    virtual double pz() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.Pz(); }
    /// transverse momentum                                                               
    virtual float pt() const GCC11_FINAL  { return ref_->pt(); } 
    /// momentum azimuthal angle                                                          
    virtual float phi() const GCC11_FINAL  { return ref_->phi(); }
    /// momentum polar angle                                                              
    virtual double theta() const GCC11_FINAL  { return ref_->theta(); }
    /// momentum pseudorapidity                                                           
    virtual float eta() const GCC11_FINAL  { return ref_->eta(); }
    /// rapidity                                                                          
    virtual double rapidity() const GCC11_FINAL  { cachePolar(); return p4Polar_.Rapidity(); }
    /// rapidity                                                                          
    virtual double y() const GCC11_FINAL  { return rapidity(); }

    /// set particle mass                                                                 
    virtual void setMass( double m ) GCC11_FINAL  {
      mass_ = m;
      clearCache();
    }

    /// vertex position                                                                   
    virtual const Point & vertex() const GCC11_FINAL  { return ref_->vertex(); }
    /// x coordinate of vertex position                                                   
    virtual double vx() const GCC11_FINAL  { return ref_->vx(); }
    /// y coordinate of vertex position                                                   
    virtual double vy() const GCC11_FINAL  { return ref_->vy(); }
    /// z coordinate of vertex position                                                   
    virtual double vz() const GCC11_FINAL  { return ref_->vz(); }


    /// returns a clone of the Candidate object                                           
    virtual LeafRefCandidateT<T> * clone() const GCC11_FINAL  {
      return new LeafRefCandidateT<T>( *this );
    }

                     
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate                                      
    virtual bool hasMasterClone() const GCC11_FINAL  { return false; }
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate                                
    virtual const CandidateBaseRef & masterClone() const GCC11_FINAL  { 
      static CandidateBaseRef dummyRef; return dummyRef; 
    }
    /// returns true if this candidate has a ptr to a master clone.                                                    
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate                                   
    virtual bool hasMasterClonePtr() const GCC11_FINAL  { return false; }
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate                             
    virtual const CandidatePtr & masterClonePtr() const GCC11_FINAL  { 
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
      struct daughter_iterator GCC11_FINAL  {
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


    virtual bool isElectron() const GCC11_FINAL  { return false; }
    virtual bool isMuon() const GCC11_FINAL  { return false; }
    virtual bool isStandAloneMuon() const GCC11_FINAL  { return false; }
    virtual bool isGlobalMuon() const GCC11_FINAL  { return false; }
    virtual bool isTrackerMuon() const GCC11_FINAL  { return false; }
    virtual bool isCaloMuon() const GCC11_FINAL  { return false; }
    virtual bool isPhoton() const GCC11_FINAL  { return false; }
    virtual bool isConvertedPhoton() const GCC11_FINAL  { return false; }
    virtual bool isJet() const GCC11_FINAL  { return false; }

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
                                        
    virtual void setCharge( Charge q ) GCC11_FINAL  { return; }                                         
    virtual int threeCharge() const GCC11_FINAL  { return 0; }
    virtual void setThreeCharge( Charge qx3 ) GCC11_FINAL  { return; }
    virtual void setP4( const LorentzVector & p4 ) GCC11_FINAL  {
      return;
    }
    virtual void setP4( const PolarLorentzVector & p4 ) GCC11_FINAL  {
      return;
    }
    virtual void setPz( double pz ) GCC11_FINAL  {
      return;
    }                            
    virtual void setVertex( const Point & vertex ) GCC11_FINAL  { return; }
    virtual void setPdgId( int pdgId ) GCC11_FINAL  { return; }                                              
    virtual int status() const GCC11_FINAL  { return 0; }                                      
    virtual void setStatus( int status ) GCC11_FINAL  { return; }
    static const unsigned int longLivedTag;                                               
    virtual void setLongLived() GCC11_FINAL  { return; }                                      
    virtual bool longLived() const GCC11_FINAL  { return false; }
    static const unsigned int massConstraintTag;
    virtual void setMassConstraint() GCC11_FINAL  { return;}
    virtual bool massConstraint() const GCC11_FINAL  { return false; }  
    virtual double vertexChi2() const GCC11_FINAL  { return 0.; }
    virtual double vertexNdof() const GCC11_FINAL  { return 0.; }
    virtual double vertexNormalizedChi2() const GCC11_FINAL  { return 0.; }
    virtual double vertexCovariance(int i, int j) const GCC11_FINAL  { return 0.; }
    CovarianceMatrix vertexCovariance() const GCC11_FINAL  { CovarianceMatrix m; return m; }
    virtual void fillVertexCovariance(CovarianceMatrix & v) const GCC11_FINAL  { return ; }
    
  };



  template<class T>
  Candidate::const_iterator LeafRefCandidateT<T>::begin() const  { 
    return const_iterator( new const_iterator_imp_specific ); 
  }

  template<class T>
  Candidate::const_iterator LeafRefCandidateT<T>::end() const   { 
    return  const_iterator( new const_iterator_imp_specific ); 
  }

  template<class T>
  Candidate::iterator LeafRefCandidateT<T>::begin()   { 
    return iterator( new iterator_imp_specific ); 
  }
  
  template<class T>
  Candidate::iterator LeafRefCandidateT<T>::end()   { 
    return iterator( new iterator_imp_specific ); 
  }
  
  
  template<class T>
  bool LeafRefCandidateT<T>::overlap( const Candidate & o ) const  { 
    return  p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
  }
  
  
  template<class T>
  bool LeafRefCandidateT<T>::overlap( const LeafRefCandidateT & o ) const   { 
    return  ref_ == o.ref_;
  }



}

#endif
