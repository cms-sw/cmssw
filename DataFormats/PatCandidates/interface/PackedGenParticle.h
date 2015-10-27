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
    : packedPt_(0), packedY_(0), packedPhi_(0), packedM_(0), 
    p4_(nullptr), p4c_(nullptr), vertex_(0,0,0),  pdgId_(0), charge_(0) { }
    explicit PackedGenParticle( const reco::GenParticle & c)
    : p4_(new PolarLorentzVector(c.pt(), c.eta(), c.phi(), c.mass())), p4c_( new LorentzVector(*p4_)), vertex_(0,0,0), pdgId_(c.pdgId()), charge_(c.charge()), mother_(c.motherRef(0)),
      statusFlags_(c.statusFlags()) { pack(); }
    explicit PackedGenParticle( const reco::GenParticle & c, const edm::Ref<reco::GenParticleCollection> &  mother)
    : p4_(new PolarLorentzVector(c.pt(), c.eta(), c.phi(), c.mass())), p4c_(new LorentzVector(*p4_)), vertex_(0,0,0), pdgId_(c.pdgId()), charge_(c.charge()), mother_(mother),
      statusFlags_(c.statusFlags()) { pack(); }

    PackedGenParticle(const PackedGenParticle& iOther)
    : packedPt_(iOther.packedPt_), packedY_(iOther.packedY_), packedPhi_(iOther.packedPhi_), packedM_(iOther.packedM_),
      p4_(nullptr),p4c_(nullptr),
      vertex_(iOther.vertex_), dxy_(iOther.dxy_), dz_(iOther.dz_),dphi_(iOther.dphi_),
      pdgId_(iOther.pdgId_),charge_(iOther.charge_),mother_(iOther.mother_),
      statusFlags_(iOther.statusFlags_) {
      if(iOther.p4c_) {
        p4_.store( new PolarLorentzVector(*iOther.p4_) );
        p4c_.store( new LorentzVector(*iOther.p4c_) );
      }
    }

    PackedGenParticle(PackedGenParticle&& iOther)
    : packedPt_(iOther.packedPt_), packedY_(iOther.packedY_), packedPhi_(iOther.packedPhi_), packedM_(iOther.packedM_),
      p4_(nullptr),p4c_(nullptr),
      vertex_(std::move(iOther.vertex_)), dxy_(iOther.dxy_), dz_(iOther.dz_),dphi_(iOther.dphi_),
      pdgId_(iOther.pdgId_),charge_(iOther.charge_),mother_(std::move(iOther.mother_)),
      statusFlags_(iOther.statusFlags_) {
      if(iOther.p4c_) {
        p4_.store( p4_.exchange(nullptr) );
        p4c_.store( p4c_.exchange(nullptr) );
      }
    }

    PackedGenParticle& operator=(PackedGenParticle&& iOther) {
      if(this != &iOther) {
        packedPt_ = iOther.packedPt_;
        packedY_ = iOther.packedY_;
        packedPhi_ = iOther.packedPhi_;
        packedM_ = iOther.packedM_;
        if(p4c_) {
          delete p4_.exchange(iOther.p4_.exchange(nullptr));
          delete p4c_.exchange(iOther.p4c_.exchange(nullptr)) ;
        } else {
          delete p4_.exchange(nullptr);
          delete p4c_.exchange(nullptr);
        }
        vertex_=std::move(iOther.vertex_);
        dxy_ = iOther.dxy_;
        dz_ = iOther.dz_;
        dphi_ = iOther.dphi_;
        pdgId_ = iOther.pdgId_;
        charge_ = iOther.charge_;
        mother_ = std::move(iOther.mother_);
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
    virtual ~PackedGenParticle();
    /// number of daughters
    virtual size_t numberOfDaughters() const;
    /// return daughter at a given position (throws an exception)
    virtual const reco::Candidate * daughter( size_type ) const;
    /// number of mothers
    virtual size_t numberOfMothers() const;
    /// return mother at a given position (throws an exception)
    virtual const reco::Candidate * mother( size_type ) const;
    /// direct access to the mother reference (may be null)
    const reco::GenParticleRef & motherRef() const { return mother_; }

    /// return daughter at a given position (throws an exception)
    virtual reco::Candidate * daughter( size_type );
    /// return daughter with a specified role name
    virtual reco::Candidate * daughter(const std::string& s );
    /// return daughter with a specified role name                                        
    virtual const reco::Candidate * daughter(const std::string& s ) const;
    /// return the number of source Candidates                                            
    /// ( the candidates used to construct this Candidate)                                
    virtual size_t numberOfSourceCandidatePtrs() const {return 0;}
    /// return a Ptr to one of the source Candidates                                      
    /// ( the candidates used to construct this Candidate)                                
    virtual reco::CandidatePtr sourceCandidatePtr( size_type i ) const {
      return reco::CandidatePtr();
    }

    /// electric charge
    virtual int charge() const {
	return charge_;	
    }
    /// set electric charge                                                               
    virtual void setCharge( int charge) {charge_=charge;}
    /// electric charge                                                                   
    virtual int threeCharge() const {return charge()*3;}
    /// set electric charge                                                               
    virtual void setThreeCharge( int threecharge) {}
    /// four-momentum Lorentz vecto r                                                      
    virtual const LorentzVector & p4() const { if (!p4c_) unpack(); return *p4c_; }  
    /// four-momentum Lorentz vector                                                      
    virtual const PolarLorentzVector & polarP4() const { if (!p4c_) unpack(); return *p4_; }
    /// spatial momentum vector                                                           
    virtual Vector momentum() const  { if (!p4c_) unpack(); return p4c_.load()->Vect(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    virtual Vector boostToCM() const { if (!p4c_) unpack(); return p4c_.load()->BoostToCM(); }
    /// magnitude of momentum vector                                                      
    virtual double p() const { if (!p4c_) unpack(); return p4c_.load()->P(); }
    /// energy                                                                            
    virtual double energy() const { if (!p4c_) unpack(); return p4c_.load()->E(); }
   /// transverse energy   
    double et() const { return (pt()<=0) ? 0 : p4c_.load()->Et(); }
    /// transverse energy squared (use this for cuts)!
    double et2() const { return (pt()<=0) ? 0 : p4c_.load()->Et2(); }
    /// mass                                                                              
    virtual double mass() const { if (!p4c_) unpack(); return p4_.load()->M(); }
    /// mass squared                                                                      
    virtual double massSqr() const { if (!p4c_) unpack(); return p4_.load()->M()*p4_.load()->M(); }

    /// transverse mass                                                                   
    virtual double mt() const { if (!p4c_) unpack(); return p4_.load()->Mt(); }
    /// transverse mass squared                                                           
    virtual double mtSqr() const { if (!p4c_) unpack(); return p4_.load()->Mt2(); }
    /// x coordinate of momentum vector                                                   
    virtual double px() const { if (!p4c_) unpack(); return p4c_.load()->Px(); }
    /// y coordinate of momentum vector                                                   
    virtual double py() const { if (!p4c_) unpack(); return p4c_.load()->Py(); }
    /// z coordinate of momentum vector                                                   
    virtual double pz() const { if (!p4c_) unpack(); return p4c_.load()->Pz(); }
    /// transverse momentum                                                               
    virtual double pt() const { if (!p4c_) unpack(); return p4_.load()->Pt();}
    /// momentum azimuthal angle                                                          
    virtual double phi() const { if (!p4c_) unpack(); return p4_.load()->Phi(); }
    /// momentum polar angle                                                              
    virtual double theta() const { if (!p4c_) unpack(); return p4_.load()->Theta(); }
    /// momentum pseudorapidity                                                           
    virtual double eta() const { if (!p4c_) unpack(); return p4_.load()->Eta(); }
    /// rapidity                                                                          
    virtual double rapidity() const { if (!p4c_) unpack(); return p4_.load()->Rapidity(); }
    /// rapidity                                                                          
    virtual double y() const { if (!p4c_) unpack(); return p4_.load()->Rapidity(); }
    /// set 4-momentum                                                                    
    virtual void setP4( const LorentzVector & p4 ) { 
        unpack(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        *p4_ = PolarLorentzVector(p4.Pt(), p4.Eta(), p4.Phi(), p4.M());
        pack();
    }
    /// set 4-momentum                                                                    
    virtual void setP4( const PolarLorentzVector & p4 ) { 
        unpack(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        *p4_ = p4; 
        pack();
    }
    /// set particle mass                                                                 
    virtual void setMass( double m ) {
      if (!p4c_) unpack(); 
      *p4_ = PolarLorentzVector(p4_.load()->Pt(), p4_.load()->Eta(), p4_.load()->Phi(), m); 
      pack();
    }
    virtual void setPz( double pz ) {
      unpack(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      *p4c_ = LorentzVector(p4c_.load()->Px(), p4c_.load()->Py(), pz, p4c_.load()->E());
      *p4_  = PolarLorentzVector(p4c_.load()->Pt(), p4c_.load()->Eta(), p4c_.load()->Phi(), p4c_.load()->M());
      pack();
    }
    /// vertex position
    virtual const Point & vertex() const { return vertex_; }//{ if (fromPV_) return Point(0,0,0); else return Point(0,0,100); }
    /// x coordinate of vertex position                                                   
    virtual double vx() const  {  return vertex_.X(); }//{ return 0; }
    /// y coordinate of vertex position                                                   
    virtual double vy() const  {  return vertex_.Y(); }//{ return 0; }
    /// z coordinate of vertex position                                                   
    virtual double vz() const  {  return vertex_.Z(); }//{ if (fromPV_) return 0; else return 100; }
    /// set vertex                                                                        
    virtual void setVertex( const Point & vertex ) { vertex_ = vertex;  }

    enum PVAssoc { NoPV=0, PVLoose=1, PVTight=2, PVUsedInFit=3 } ;


    /// dxy with respect to the PV ref
    virtual float dxy() const { unpack(); return dxy_; }
    /// dz with respect to the PV ref
    virtual float dz()  const { unpack(); return dz_; }
    /// dxy with respect to another point
    virtual float dxy(const Point &p) const ;
    /// dz  with respect to another point
    virtual float dz(const Point &p)  const ;


    /// PDG identifier                                                                    
    virtual int pdgId() const   { return pdgId_; }
    // set PDG identifier                                                                 
    virtual void setPdgId( int pdgId )   { pdgId_ = pdgId; }
    /// status word                                                                       
    virtual int status() const   { return 1; } /*FIXME*/
    /// set status word                                                                   
    virtual void setStatus( int status ) {} /*FIXME*/
    /// long lived flag                                                                   
    static const unsigned int longLivedTag = 0; /*FIXME*/
    /// set long lived flag                                                               
    virtual void setLongLived() {} /*FIXME*/
    /// is long lived?                                                                    
    virtual bool longLived() const;
    /// do mass constraint flag
    static const unsigned int massConstraintTag = 0; /*FIXME*/ 
    /// set mass constraint flag
    virtual void setMassConstraint() {} /*FIXME*/
    /// do mass constraint?
    virtual bool massConstraint() const;

    /// returns a clone of the Candidate object                                           
    virtual PackedGenParticle * clone() const  {
      return new PackedGenParticle( *this );
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
    CovarianceMatrix vertexCovariance() const   { CovarianceMatrix m; fillVertexCovariance(m); return m; }
    /// fill SMatrix                                                                                                   
    virtual void fillVertexCovariance(CovarianceMatrix & v) const;
    /// returns true if this candidate has a reference to a master clone.                                              
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate                                      
    virtual bool hasMasterClone() const;
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate                                
    virtual const reco::CandidateBaseRef & masterClone() const;
    /// returns true if this candidate has a ptr to a master clone.                                                    
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate                                   
    virtual bool hasMasterClonePtr() const;
    /// returns ptr to master clone, if existing.                                                                      
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate                             

    virtual const reco::CandidatePtr & masterClonePtr() const;

    /// cast master clone reference to a concrete type
    template<typename Ref>
      Ref masterRef() const { return masterClone().template castTo<Ref>(); }
    /// get a component

    virtual bool isElectron() const;
    virtual bool isMuon() const;
    virtual bool isStandAloneMuon() const;
    virtual bool isGlobalMuon() const;
    virtual bool isTrackerMuon() const;
    virtual bool isCaloMuon() const;
    virtual bool isPhoton() const;
    virtual bool isConvertedPhoton() const;
    virtual bool isJet() const;
    
    const reco::GenStatusFlags &statusFlags() const { return statusFlags_; }
    reco::GenStatusFlags &statusFlags() { return statusFlags_; }
    
    /////////////////////////////////////////////////////////////////////////////
    //basic set of gen status flags accessible directly here
    //the rest accessible through statusFlags()
    //(see GenStatusFlags.h for their meaning)
    
    /////////////////////////////////////////////////////////////////////////////
    //these are robust, generator-independent functions for categorizing
    //mainly final state particles, but also intermediate hadrons/taus
    
    //is particle prompt (not from hadron, muon, or tau decay) and final state
    bool isPromptFinalState() const { return status()==1 && statusFlags_.isPrompt(); }
        
    //this particle is a direct decay product of a prompt tau and is final state
    //(eg an electron or muon from a leptonic decay of a prompt tau)
    bool isDirectPromptTauDecayProductFinalState() const { return status()==1 && statusFlags_.isDirectPromptTauDecayProduct(); }
    
    /////////////////////////////////////////////////////////////////////////////
    //these are generator history-dependent functions for tagging particles
    //associated with the hard process
    //Currently implemented for Pythia 6 and Pythia 8 status codes and history   
    //and may not have 100% consistent meaning across all types of processes
    //Users are strongly encouraged to stick to the more robust flags above,
    //as well as the expanded set available in GenStatusFlags.h
    
    //this particle is the final state direct descendant of a hard process particle  
    bool fromHardProcessFinalState() const { return status()==1 && statusFlags_.fromHardProcess(); }
        
    //this particle is a direct decay product of a hardprocess tau and is final state
    //(eg an electron or muon from a leptonic decay of a tau from the hard process)
    bool isDirectHardProcessTauDecayProductFinalState() const { return status()==1 && statusFlags_.isDirectHardProcessTauDecayProduct(); }
        

  protected:
    uint16_t packedPt_, packedY_, packedPhi_, packedM_;
    void pack(bool unpackAfterwards=true) ;
    void unpack() const ;
 
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
    virtual bool overlap( const reco::Candidate & ) const;
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  };

  typedef std::vector<pat::PackedGenParticle> PackedGenParticleCollection;
  typedef edm::Ref<pat::PackedGenParticleCollection> PackedGenParticleRef;
  typedef edm::RefVector<pat::PackedGenParticleCollection> PackedGenParticleRefVector;
}

#endif
