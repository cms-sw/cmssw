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
#include "ParticleState.h"

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

    LeafCandidate() {}

    // constructor from candidate 
    explicit LeafCandidate( const Candidate & c) : m_state(c.charge(),c.polarP4(), c.vertex(), c.pdgId(), c.status() ){}

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    template<typename... Args>
    explicit   LeafCandidate(Args && ...args) : 
    m_state(std::forward<Args>(args)...) {}

    LeafCandidate(LeafCandidate& rh): m_state(rh.m_state){}

    LeafCandidate(LeafCandidate&&)=default;
    LeafCandidate(LeafCandidate const&)=default;
    LeafCandidate& operator=(LeafCandidate&&)=default;
    LeafCandidate& operator=(LeafCandidate const&)=default;
#else
    // for Reflex to parse...  (compilation will use the above)
    LeafCandidate( Charge q, const PtEtaPhiMass & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
    LeafCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
    LeafCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
    LeafCandidate( Charge q, const GlobalVector & p3, float iEnergy, float imass, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
#endif

    void construct(int qx3,  float pt, float eta, float phi, float mass, const Point & vtx, int pdgId, int status) {
      m_state = ParticleState(qx3, PolarLorentzVector(pt,eta,phi,mass), vtx, pdgId, status, false);
    }

    /// destructor
    virtual ~LeafCandidate();
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
    virtual int charge() const GCC11_FINAL { return m_state.charge(); }
    /// set electric charge                                                               
    virtual void setCharge( Charge q ) GCC11_FINAL { m_state.setCharge(q); }
    /// electric charge                                                                   
    virtual int threeCharge() const GCC11_FINAL { return m_state.threeCharge(); }
    /// set electric charge                                                               
    virtual void setThreeCharge( Charge qx3 ) GCC11_FINAL {m_state.setThreeCharge(qx3); }
    /// four-momentum Lorentz vector                                                      
    virtual const LorentzVector & p4() const GCC11_FINAL { return m_state.p4(); }
    /// four-momentum Lorentz vector                                                      
    virtual const PolarLorentzVector & polarP4() const GCC11_FINAL { return m_state.polarP4(); }
    /// spatial momentum vector                                                           
    virtual Vector momentum() const GCC11_FINAL { return m_state.momentum(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    virtual Vector boostToCM() const GCC11_FINAL { return m_state.boostToCM(); }
    /// magnitude of momentum vector                                                      
    virtual double p() const GCC11_FINAL { return m_state.p(); }
    /// energy                                                                            
    virtual double energy() const GCC11_FINAL { return m_state.energy(); }
    /// transverse energy                                                                 
    virtual double et() const GCC11_FINAL { return m_state.et(); }
    /// transverse energy squared (use this for cut!)                                                                 
    virtual double et2() const GCC11_FINAL { return m_state.et2(); }
    /// mass                                                                              
    virtual double mass() const GCC11_FINAL { return m_state.mass(); }
    /// mass squared                                                                      
    virtual double massSqr() const GCC11_FINAL { return mass() * mass(); }

    /// transverse mass                                                                   
    virtual double mt() const GCC11_FINAL  { return m_state.mt(); }
    /// transverse mass squared                                                           
    virtual double mtSqr() const GCC11_FINAL  { return m_state.mtSqr(); }
    /// x coordinate of momentum vector                                                   
    virtual double px() const GCC11_FINAL  {  return m_state.px(); }
    /// y coordinate of momentum vector                                                   
    virtual double py() const GCC11_FINAL  { return m_state.py(); }
    /// z coordinate of momentum vector                                                   
    virtual double pz() const GCC11_FINAL  {  return m_state.pz(); }
    /// transverse momentum                                                               
    virtual double pt() const GCC11_FINAL  { return m_state.pt();}
    /// momentum azimuthal angle                                                          
    virtual double phi() const GCC11_FINAL  { return m_state.phi(); }
    /// momentum polar angle                                                              
    virtual double theta() const GCC11_FINAL  {  return m_state.theta(); }
    /// momentum pseudorapidity                                                           
    virtual  double eta() const GCC11_FINAL  { return m_state.eta(); }
    /// rapidity                                                                          
    virtual double rapidity() const GCC11_FINAL  {  return m_state.rapidity(); }
    /// rapidity                                                                          
    virtual double y() const GCC11_FINAL  { return rapidity(); }
    /// set 4-momentum                                                                    
    virtual void setP4( const LorentzVector & p4 ) GCC11_FINAL  { m_state.setP4(p4);}
    /// set 4-momentum                                                                    
    virtual void setP4( const PolarLorentzVector & p4 ) GCC11_FINAL  {m_state.setP4(p4); }
    /// set particle mass                                                                 
    virtual void setMass( double m ) GCC11_FINAL  {m_state.setMass(m);}
    virtual void setPz( double pz ) GCC11_FINAL  { m_state.setPz(pz);}
    /// vertex position                 (overwritten by PF...)                                                  
    virtual const Point & vertex() const { return m_state.vertex(); }
    /// x coordinate of vertex position                                                   
    virtual double vx() const  { return m_state.vx(); }
    /// y coordinate of vertex position                                                   
    virtual double vy() const  { return m_state.vy(); }
    /// z coordinate of vertex position                                                   
    virtual double vz() const  { return m_state.vz(); }
    /// set vertex                                                                        
    virtual void setVertex( const Point & vertex )   { m_state.setVertex(vertex); }

    /// PDG identifier                                                                    
    virtual int pdgId() const GCC11_FINAL  { return m_state.pdgId(); }
    // set PDG identifier                                                                 
    virtual void setPdgId( int pdgId ) GCC11_FINAL  { m_state.setPdgId(pdgId); }
    /// status word                                                                       
    virtual int status() const GCC11_FINAL  { return m_state.status(); }
    /// set status word                                                                   
    virtual void setStatus( int status ) GCC11_FINAL  { m_state.setStatus(status); }
    /// long lived flag                                                                   
    /// set long lived flag                                                               
    virtual void setLongLived() GCC11_FINAL  { m_state.setLongLived(); }
    /// is long lived?                                                                    
    virtual bool longLived() const GCC11_FINAL  { return m_state.longLived(); }
    /// do mass constraint flag
    /// set mass constraint flag
    virtual void setMassConstraint() GCC11_FINAL  { m_state.setMassConstraint();}
    /// do mass constraint?
    virtual bool massConstraint() const GCC11_FINAL  { return m_state.massConstraint(); }

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



    virtual bool isElectron() const;
    virtual bool isMuon() const;
    virtual bool isStandAloneMuon() const;
    virtual bool isGlobalMuon() const;
    virtual bool isTrackerMuon() const;
    virtual bool isCaloMuon() const;
    virtual bool isPhoton() const;
    virtual bool isConvertedPhoton() const;
    virtual bool isJet() const;

  private:
    ParticleState m_state;

    private:
    /// check overlap with another Candidate                                              
    virtual bool overlap( const Candidate & ) const;
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  };

}

#endif
