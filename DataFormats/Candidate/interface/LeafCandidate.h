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
    ~LeafCandidate() override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// return daughter at a given position (throws an exception)
    const Candidate * daughter( size_type ) const override;
    /// number of mothers
    size_t numberOfMothers() const override;
    /// return mother at a given position (throws an exception)
    const Candidate * mother( size_type ) const override;
    /// return daughter at a given position (throws an exception)
    Candidate * daughter( size_type ) override;
    /// return daughter with a specified role name
    Candidate * daughter(const std::string& s ) override;
    /// return daughter with a specified role name                                        
    const Candidate * daughter(const std::string& s ) const override;
    /// return the number of source Candidates                                            
    /// ( the candidates used to construct this Candidate)                                
    size_t numberOfSourceCandidatePtrs() const override { return 0;}
    /// return a Ptr to one of the source Candidates                                      
    /// ( the candidates used to construct this Candidate)                                
    CandidatePtr sourceCandidatePtr( size_type i ) const override {
      return CandidatePtr();
    }

    /// electric charge
    int charge() const final { return m_state.charge(); }
    /// set electric charge                                                               
    void setCharge( Charge q ) final { m_state.setCharge(q); }
    /// electric charge                                                                   
    int threeCharge() const final { return m_state.threeCharge(); }
    /// set electric charge                                                               
    void setThreeCharge( Charge qx3 ) final {m_state.setThreeCharge(qx3); }
    /// four-momentum Lorentz vector                                                      
    const LorentzVector & p4() const final { return m_state.p4(); }
    /// four-momentum Lorentz vector                                                      
    const PolarLorentzVector & polarP4() const final { return m_state.polarP4(); }
    /// spatial momentum vector                                                           
    Vector momentum() const final { return m_state.momentum(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    Vector boostToCM() const final { return m_state.boostToCM(); }
    /// magnitude of momentum vector                                                      
    double p() const final { return m_state.p(); }
    /// energy                                                                            
    double energy() const final { return m_state.energy(); }
    /// transverse energy                                                                 
    double et() const final { return m_state.et(); }
    /// transverse energy squared (use this for cut!)                                                                 
    double et2() const final { return m_state.et2(); }
    /// mass                                                                              
    double mass() const final { return m_state.mass(); }
    /// mass squared                                                                      
    double massSqr() const final { return mass() * mass(); }

    /// transverse mass                                                                   
    double mt() const final  { return m_state.mt(); }
    /// transverse mass squared                                                           
    double mtSqr() const final  { return m_state.mtSqr(); }
    /// x coordinate of momentum vector                                                   
    double px() const final  {  return m_state.px(); }
    /// y coordinate of momentum vector                                                   
    double py() const final  { return m_state.py(); }
    /// z coordinate of momentum vector                                                   
    double pz() const final  {  return m_state.pz(); }
    /// transverse momentum                                                               
    double pt() const final  { return m_state.pt();}
    /// momentum azimuthal angle                                                          
    double phi() const final  { return m_state.phi(); }
    /// momentum polar angle                                                              
    double theta() const final  {  return m_state.theta(); }
    /// momentum pseudorapidity                                                           
     double eta() const final  { return m_state.eta(); }
    /// rapidity                                                                          
    double rapidity() const final  {  return m_state.rapidity(); }
    /// rapidity                                                                          
    double y() const final  { return rapidity(); }
    /// set 4-momentum                                                                    
    void setP4( const LorentzVector & p4 ) final  { m_state.setP4(p4);}
    /// set 4-momentum                                                                    
    void setP4( const PolarLorentzVector & p4 ) final  {m_state.setP4(p4); }
    /// set particle mass                                                                 
    void setMass( double m ) final  {m_state.setMass(m);}
    void setPz( double pz ) final  { m_state.setPz(pz);}
    /// vertex position                 (overwritten by PF...)                                                  
    const Point & vertex() const override { return m_state.vertex(); }
    /// x coordinate of vertex position                                                   
    double vx() const override  { return m_state.vx(); }
    /// y coordinate of vertex position                                                   
    double vy() const override  { return m_state.vy(); }
    /// z coordinate of vertex position                                                   
    double vz() const override  { return m_state.vz(); }
    /// set vertex                                                                        
    void setVertex( const Point & vertex ) override   { m_state.setVertex(vertex); }

    /// PDG identifier                                                                    
    int pdgId() const final  { return m_state.pdgId(); }
    // set PDG identifier                                                                 
    void setPdgId( int pdgId ) final  { m_state.setPdgId(pdgId); }
    /// status word                                                                       
    int status() const final  { return m_state.status(); }
    /// set status word                                                                   
    void setStatus( int status ) final  { m_state.setStatus(status); }
    /// long lived flag                                                                   
    /// set long lived flag                                                               
    void setLongLived() final  { m_state.setLongLived(); }
    /// is long lived?                                                                    
    bool longLived() const final  { return m_state.longLived(); }
    /// do mass constraint flag
    /// set mass constraint flag
    void setMassConstraint() final  { m_state.setMassConstraint();}
    /// do mass constraint?
    bool massConstraint() const final  { return m_state.massConstraint(); }

    /// returns a clone of the Candidate object                                           
    LeafCandidate * clone() const override  {
      return new LeafCandidate( *this );
    }

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
    CovarianceMatrix vertexCovariance() const final  { CovarianceMatrix m; fillVertexCovariance(m); return m; }
    /// fill SMatrix                                                                                                   
    void fillVertexCovariance(CovarianceMatrix & v) const override;
    /// returns true if this candidate has a reference to a master clone.                                              
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate                                      
    bool hasMasterClone() const override;
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate                                
    const CandidateBaseRef & masterClone() const override;
    /// returns true if this candidate has a ptr to a master clone.                                                    
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate                                   
    bool hasMasterClonePtr() const override;
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate                             
    const CandidatePtr & masterClonePtr() const override;

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



    bool isElectron() const override;
    bool isMuon() const override;
    bool isStandAloneMuon() const override;
    bool isGlobalMuon() const override;
    bool isTrackerMuon() const override;
    bool isCaloMuon() const override;
    bool isPhoton() const override;
    bool isConvertedPhoton() const override;
    bool isJet() const override;

  private:
    ParticleState m_state;

    private:
    /// check overlap with another Candidate                                              
    bool overlap( const Candidate & ) const override;
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  };

}

#endif
