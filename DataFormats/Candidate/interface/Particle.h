#ifndef Candidate_Particle_h
#define Candidate_Particle_h
/** \class reco::Particle
 *
 * Base class describing a generic reconstructed particle
 * its main subclass is Candidate
 *
 * \author Luca Lista, INFN
 *
 *
 */

#include "ParticleState.h"
namespace reco {
  
  class Particle {
  public:
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

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    template<typename... Args>
    explicit   Particle(Args && ...args) : 
    m_state(std::forward<Args>(args)...) {}

    Particle(Particle& rh): m_state(rh.m_state){}

    Particle(Particle&&)=default;
    Particle(Particle const&)=default;
    Particle& operator=(Particle&&)=default;
    Particle& operator=(Particle const&)=default;
#else
    // for Reflex to parse...  (compilation will use the above)
    Particle();
    Particle( Charge q, const PtEtaPhiMass & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
    Particle( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
    Particle( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
    Particle( Charge q, const GlobalVector & p3, float iEnergy, float imass, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true );
#endif

    void construct(int qx3,  float pt, float eta, float phi, float mass, const Point & vtx, int pdgId, int status) {
      m_state = ParticleState(qx3, PolarLorentzVector(pt,eta,phi,mass), vtx, pdgId, status, false);
    }

    /// destructor
    virtual ~Particle(){}


    /// electric charge
    int charge() const { return m_state.charge(); }
    /// set electric charge                                                               
    void setCharge( Charge q ) { m_state.setCharge(q); }
    /// electric charge                                                                   
    int threeCharge() const { return m_state.threeCharge(); }
    /// set electric charge                                                               
    void setThreeCharge( Charge qx3 ) {m_state.setThreeCharge(qx3); }
    /// four-momentum Lorentz vector                                                      
    const LorentzVector & p4() const { return m_state.p4(); }
    /// four-momentum Lorentz vector                                                      
    const PolarLorentzVector & polarP4() const { return m_state.polarP4(); }
    /// spatial momentum vector                                                           
    Vector momentum() const { return m_state.momentum(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    Vector boostToCM() const { return m_state.boostToCM(); }
    /// magnitude of momentum vector                                                      
    double p() const { return m_state.p(); }
    /// energy                                                                            
    double energy() const { return m_state.energy(); }
    /// transverse energy                                                                 
    double et() const { return m_state.et(); }
    /// transverse energy squared (use this for cut!)                                                                 
    double et2() const { return m_state.et2(); }
    /// mass                                                                              
    double mass() const { return m_state.mass(); }
    /// mass squared                                                                      
    double massSqr() const { return mass() * mass(); }

    /// transverse mass                                                                   
    double mt() const  { return m_state.mt(); }
    /// transverse mass squared                                                           
    double mtSqr() const  { return m_state.mtSqr(); }
    /// x coordinate of momentum vector                                                   
    double px() const  {  return m_state.px(); }
    /// y coordinate of momentum vector                                                   
    double py() const  { return m_state.py(); }
    /// z coordinate of momentum vector                                                   
    double pz() const  {  return m_state.pz(); }
    /// transverse momentum                                                               
    double pt() const  { return m_state.pt();}
    /// momentum azimuthal angle                                                          
    double phi() const  { return m_state.phi(); }
    /// momentum polar angle                                                              
    double theta() const  {  return m_state.theta(); }
    /// momentum pseudorapidity                                                           
     double eta() const  { return m_state.eta(); }
    /// rapidity                                                                          
    double rapidity() const  {  return m_state.rapidity(); }
    /// rapidity                                                                          
    double y() const  { return rapidity(); }
    /// set 4-momentum                                                                    
    void setP4( const LorentzVector & p4 )  { m_state.setP4(p4);}
    /// set 4-momentum                                                                    
    void setP4( const PolarLorentzVector & p4 )  {m_state.setP4(p4); }
    /// set particle mass                                                                 
    void setMass( double m )  {m_state.setMass(m);}
    void setPz( double pz )  { m_state.setPz(pz);}
    /// vertex position                 (overwritten by PF...)                                                  
    const Point & vertex() const { return m_state.vertex(); }
    /// x coordinate of vertex position                                                   
    double vx() const  { return m_state.vx(); }
    /// y coordinate of vertex position                                                   
    double vy() const  { return m_state.vy(); }
    /// z coordinate of vertex position                                                   
    double vz() const  { return m_state.vz(); }
    /// set vertex                                                                        
    void setVertex( const Point & vertex )   { m_state.setVertex(vertex); }

    /// PDG identifier                                                                    
    int pdgId() const  { return m_state.pdgId(); }
    // set PDG identifier                                                                 
    void setPdgId( int pdgId )  { m_state.setPdgId(pdgId); }
    /// status word                                                                       
    int status() const  { return m_state.status(); }
    /// set status word                                                                   
    void setStatus( int status )  { m_state.setStatus(status); }
    /// long lived flag                                                                   
    /// set long lived flag                                                               
    void setLongLived()  { m_state.setLongLived(); }
    /// is long lived?                                                                    
    bool longLived() const  { return m_state.longLived(); }
    /// do mass constraint flag
    /// set mass constraint flag
    void setMassConstraint()  { m_state.setMassConstraint();}
    /// do mass constraint?
    bool massConstraint() const  { return m_state.massConstraint(); }

  private:
    ParticleState m_state;


  };

}

#endif
