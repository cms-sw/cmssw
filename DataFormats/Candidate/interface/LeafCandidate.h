#ifndef Candidate_LeafCandidate_h
#define Candidate_LeafCandidate_h
/** \class reco::LeafCandidate
 *
 * particle candidate with no constituent nor daughters
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

#include "DataFormats/Math/interface/PtEtaPhiMass.h"

namespace reco {
  
  class LeafCandidate : public Candidate {
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
    LeafCandidate() : 
      qx3_(0), pt_(0), eta_(0), phi_(0), mass_(0), 
      vertex_(0, 0, 0), pdgId_(0), status_(0),
      cachePolarFixed_( false ), cacheCartesianFixed_( false ) { }
    // constructor from candidate                                                         
    explicit LeafCandidate( const Candidate & c) :
    qx3_( c.charge()*3 ), pt_( c.p4().pt() ), eta_( c.p4().eta() ), phi_( c.p4().phi() )
    , mass_( c.p4().mass() ),
    vertex_( c.vertex() ), pdgId_( c.pdgId() ), status_( c.status() ),
    cachePolarFixed_( false ), cacheCartesianFixed_( false ) {}
    
    /// constructor from Any values
    template<typename P4>
    LeafCandidate( Charge q, const P4 & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      qx3_(integerCharge ? 3*q : q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
      vertex_( vtx ), pdgId_( pdgId ), status_( status ),
      cachePolarFixed_( false ), cacheCartesianFixed_( false ) {}


    /// constructor from values  
    LeafCandidate( Charge q, const PtEtaPhiMass & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      qx3_(integerCharge ? 3*q : q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
      vertex_( vtx ), pdgId_( pdgId ), status_( status ),
      cachePolarFixed_( false ), cacheCartesianFixed_( false ) {}
   
    /// constructor from values  
    LeafCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      qx3_( q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
      vertex_( vtx ), pdgId_( pdgId ), status_( status ),
      cachePolarFixed_( false ), cacheCartesianFixed_( false ) {
      if ( integerCharge ) qx3_ *= 3;
    }
    /// constructor from values                                                           
    LeafCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      qx3_( q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
      vertex_( vtx ), pdgId_( pdgId ), status_( status ),
      cachePolarFixed_( false ), cacheCartesianFixed_( false ){
      if ( integerCharge ) qx3_ *= 3;
    }
    
    /// destructor
    virtual ~LeafCandidate();
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter iterator
    virtual iterator end();
    /// number of daughters
    virtual size_t numberOfDaughters() const;
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * daughter( size_type ) const;
    /// number of mothers
    virtual size_t numberOfMothers() const;
    /// return mother at a given position (throws an exception)
    virtual const Candidate * mother( size_type ) const;
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type );
    /// return daughter with a specified role name
    virtual Candidate * daughter(const std::string& s );
    /// return daughter with a specified role name                                        
    virtual const Candidate * daughter(const std::string& s ) const;
    /// return the number of source Candidates                                            
    /// ( the candidates used to construct this Candidate)                                
    virtual size_t numberOfSourceCandidatePtrs() const { return 0;}
    /// return a Ptr to one of the source Candidates                                      
    /// ( the candidates used to construct this Candidate)                                
    virtual CandidatePtr sourceCandidatePtr( size_type i ) const {
      return CandidatePtr();
    }

    /// electric charge
    virtual int charge() const GCC11_FINAL { return qx3_ / 3; }
    /// set electric charge                                                               
    virtual void setCharge( Charge q ) GCC11_FINAL { qx3_ = q * 3; }
    /// electric charge                                                                   
    virtual int threeCharge() const GCC11_FINAL { return qx3_; }
    /// set electric charge                                                               
    virtual void setThreeCharge( Charge qx3 ) GCC11_FINAL { qx3_ = qx3; }
    /// four-momentum Lorentz vector                                                      
    virtual const LorentzVector & p4() const GCC11_FINAL { cacheCartesian(); return p4Cartesian_; }
    /// four-momentum Lorentz vector                                                      
    virtual const PolarLorentzVector & polarP4() const GCC11_FINAL { cachePolar(); return p4Polar_; }
    /// spatial momentum vector                                                           
    virtual Vector momentum() const GCC11_FINAL { cacheCartesian(); return p4Cartesian_.Vect(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    virtual Vector boostToCM() const GCC11_FINAL { cacheCartesian(); return p4Cartesian_.BoostToCM(); }
    /// magnitude of momentum vector                                                      
    virtual double p() const GCC11_FINAL { cacheCartesian(); return p4Cartesian_.P(); }
    /// energy                                                                            
    virtual double energy() const GCC11_FINAL { cacheCartesian(); return p4Cartesian_.E(); }
    /// transverse energy                                                                 
    virtual double et() const GCC11_FINAL { cachePolar(); return p4Polar_.Et(); }
    /// mass                                                                              
    virtual float mass() const GCC11_FINAL { return mass_; }
    /// mass squared                                                                      
    virtual float massSqr() const GCC11_FINAL { return mass_ * mass_; }

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
    virtual float pt() const GCC11_FINAL  { return pt_;}
    /// momentum azimuthal angle                                                          
    virtual float phi() const GCC11_FINAL  { return phi_; }
    /// momentum polar angle                                                              
    virtual double theta() const GCC11_FINAL  { cacheCartesian(); return p4Cartesian_.Theta(); }
    /// momentum pseudorapidity                                                           
    virtual float eta() const GCC11_FINAL  { return eta_; }
    /// rapidity                                                                          
    virtual double rapidity() const GCC11_FINAL  { cachePolar(); return p4Polar_.Rapidity(); }
    /// rapidity                                                                          
    virtual double y() const GCC11_FINAL  { return rapidity(); }
    /// set 4-momentum                                                                    
    virtual void setP4( const LorentzVector & p4 ) GCC11_FINAL  {
      p4Cartesian_ = p4;
      p4Polar_ = p4;
      pt_ = p4Polar_.pt();
      eta_ = p4Polar_.eta();
      phi_ = p4Polar_.phi();
      mass_ = p4Polar_.mass();
      cachePolarFixed_ = true;
      cacheCartesianFixed_ = true;
    }
    /// set 4-momentum                                                                    
    virtual void setP4( const PolarLorentzVector & p4 ) GCC11_FINAL  {
      p4Polar_ = p4;
      pt_ = p4Polar_.pt();
      eta_ = p4Polar_.eta();
      phi_ = p4Polar_.phi();
      mass_ = p4Polar_.mass();
      cachePolarFixed_ = true;
      cacheCartesianFixed_ = false;
    }
    /// set particle mass                                                                 
    virtual void setMass( double m ) GCC11_FINAL  {
      mass_ = m;
      clearCache();
    }
    virtual void setPz( double pz ) GCC11_FINAL  {
      cacheCartesian();
      p4Cartesian_.SetPz(pz);
      p4Polar_ = p4Cartesian_;
      pt_ = p4Polar_.pt();
      eta_ = p4Polar_.eta();
      phi_ = p4Polar_.phi();
      mass_ = p4Polar_.mass();
    }
    /// vertex position                 (overwritten by PF...)                                                  
    virtual const Point & vertex() const { return vertex_; }
    /// x coordinate of vertex position                                                   
    virtual double vx() const  { return vertex_.X(); }
    /// y coordinate of vertex position                                                   
    virtual double vy() const  { return vertex_.Y(); }
    /// z coordinate of vertex position                                                   
    virtual double vz() const  { return vertex_.Z(); }
    /// set vertex                                                                        
    virtual void setVertex( const Point & vertex )   { vertex_ = vertex; }

    /// PDG identifier                                                                    
    virtual int pdgId() const GCC11_FINAL  { return pdgId_; }
    // set PDG identifier                                                                 
    virtual void setPdgId( int pdgId ) GCC11_FINAL  { pdgId_ = pdgId; }
    /// status word                                                                       
    virtual int status() const GCC11_FINAL  { return status_; }
    /// set status word                                                                   
    virtual void setStatus( int status ) GCC11_FINAL  { status_ = status; }
    /// long lived flag                                                                   
    static const unsigned int longLivedTag;
    /// set long lived flag                                                               
    virtual void setLongLived() GCC11_FINAL  { status_ |= longLivedTag; }
    /// is long lived?                                                                    
    virtual bool longLived() const GCC11_FINAL  { return status_ & longLivedTag; }
    /// do mass constraint flag
    static const unsigned int massConstraintTag;
    /// set mass constraint flag
    virtual void setMassConstraint() GCC11_FINAL  { status_ |= massConstraintTag;}
    /// do mass constraint?
    virtual bool massConstraint() const GCC11_FINAL  { return status_ & massConstraintTag; }

    /// returns a clone of the Candidate object                                           
    virtual LeafCandidate * clone() const  {
      return new LeafCandidate( *this );
    }

    /// chi-squares                                                                                                    
    virtual double vertexChi2() const;
    /** Number of degrees of freedom                                                                                   
     *  Meant to be Double32_t for soft-assignment fitters:                                                            
     *  tracks may contribute to the vertex with fractional weights.                                                   
     *  The ndof is then = to the sum of the track weights.                                                            
     *  see e.g. CMS NOTE-2006/032, CMS NOTE-2004/002                                                                  
     */
    virtual double vertexNdof() const;
    /// chi-squared divided by n.d.o.f.                                                                                
    virtual double vertexNormalizedChi2() const;
    /// (i, j)-th element of error matrix, i, j = 0, ... 2                                                             
    virtual double vertexCovariance(int i, int j) const;
    /// return SMatrix                                                                                                 
    CovarianceMatrix vertexCovariance() const GCC11_FINAL  { CovarianceMatrix m; fillVertexCovariance(m); return m; }
    /// fill SMatrix                                                                                                   
    virtual void fillVertexCovariance(CovarianceMatrix & v) const;
    /// returns true if this candidate has a reference to a master clone.                                              
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate                                      
    virtual bool hasMasterClone() const;
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate                                
    virtual const CandidateBaseRef & masterClone() const;
    /// returns true if this candidate has a ptr to a master clone.                                                    
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate                                   
    virtual bool hasMasterClonePtr() const;
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate                             
    virtual const CandidatePtr & masterClonePtr() const;

    /// cast master clone reference to a concrete type                                                                 
    template<typename Ref>
      Ref masterRef() const { return masterClone().template castTo<Ref>(); }
    /// get a component                                                                                                

    template<typename T> T get() const {
      if ( hasMasterClone() ) return masterClone()->get<T>();
      else return reco::get<T>( * this );
    }
    /// get a component                                                                                                
    template<typename T, typename Tag> T get() const {
      if ( hasMasterClone() ) return masterClone()->get<T, Tag>();
      else return reco::get<T, Tag>( * this );
    }
    /// get a component                                                                                                
    template<typename T> T get( size_type i ) const {
      if ( hasMasterClone() ) return masterClone()->get<T>( i );
      else return reco::get<T>( * this, i );
    }
    /// get a component                                                                                                
    template<typename T, typename Tag> T get( size_type i ) const {
      if ( hasMasterClone() ) return masterClone()->get<T, Tag>( i );
      else return reco::get<T, Tag>( * this, i );
    }
    /// number of components                                                                                           
    template<typename T> size_type numberOf() const {
      if ( hasMasterClone() ) return masterClone()->numberOf<T>();
      else return reco::numberOf<T>( * this );
    }
    /// number of components                                                                                           
    template<typename T, typename Tag> size_type numberOf() const {
      if ( hasMasterClone() ) return masterClone()->numberOf<T, Tag>();
      else return reco::numberOf<T, Tag>( * this );
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


    virtual bool isElectron() const;
    virtual bool isMuon() const;
    virtual bool isStandAloneMuon() const;
    virtual bool isGlobalMuon() const;
    virtual bool isTrackerMuon() const;
    virtual bool isCaloMuon() const;
    virtual bool isPhoton() const;
    virtual bool isConvertedPhoton() const;
    virtual bool isJet() const;

  protected:
    /// electric charge                                                                   
    Charge qx3_;
    /// four-momentum Lorentz vector                                                      
    float pt_, eta_, phi_, mass_;
    /// vertex position                                                                   
    Point vertex_;
    /// PDG identifier                                                                    
    int pdgId_;
    /// status word                                                                       
    int status_;
    /// internal cache for p4                                                             
    mutable PolarLorentzVector p4Polar_;
    /// internal cache for p4                                                             
    mutable LorentzVector p4Cartesian_;
    /// has cache been set?                                                               
    mutable  bool cachePolarFixed_, cacheCartesianFixed_;
    /// set internal cache                                                                
    inline void cachePolar() const {
      if ( cachePolarFixed_ ) return;
      p4Polar_ = PolarLorentzVector( pt_, eta_, phi_, mass_ );
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
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  private:
    // const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    // iterator implementation
    typedef candidate::iterator_imp_specific<daughters> iterator_imp_specific;
  };

}

#endif
