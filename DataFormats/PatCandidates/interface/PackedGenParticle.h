#ifndef __AnalysisDataFormats_PackedGenParticle_h__
#define __AnalysisDataFormats_PackedGenParticle_h__

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/iterator_imp_specific.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h" 
/* #include "DataFormats/Math/interface/PtEtaPhiMass.h" */

namespace pat {
  class PackedGenParticle : public reco::Candidate {
  public:
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
    : p4_(0,0,0,0), p4c_(0,0,0,0), vertex_(0,0,0),  pdgId_(0), charge_(0), unpacked_(false) { }
  explicit PackedGenParticle( const reco::GenParticle & c)
    : p4_(c.pt(), c.eta(), c.phi(), c.mass()), p4c_(p4_), vertex_(0,0,0), pdgId_(c.pdgId()), charge_(c.charge()), mother_(c.motherRef(0)), unpacked_(true)  { pack(); }
  explicit PackedGenParticle( const reco::GenParticle & c, const edm::Ref<reco::GenParticleCollection> &  mother)
    : p4_(c.pt(), c.eta(), c.phi(), c.mass()), p4c_(p4_), vertex_(0,0,0), pdgId_(c.pdgId()), charge_(c.charge()), mother_(mother), unpacked_(true)  { pack(); }

    
    /// destructor
    virtual ~PackedGenParticle();
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
    virtual const reco::Candidate * daughter( size_type ) const;
    /// number of mothers
    virtual size_t numberOfMothers() const;
    /// return mother at a given position (throws an exception)
    virtual const reco::Candidate * mother( size_type ) const;
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
    virtual const LorentzVector & p4() const { if (!unpacked_) unpack(); return p4c_; }  
    /// four-momentum Lorentz vector                                                      
    virtual const PolarLorentzVector & polarP4() const { if (!unpacked_) unpack(); return p4_; }
    /// spatial momentum vector                                                           
    virtual Vector momentum() const  { if (!unpacked_) unpack(); return p4c_.Vect(); }
    /// boost vector to boost a Lorentz vector                                            
    /// to the particle center of mass system                                             
    virtual Vector boostToCM() const { if (!unpacked_) unpack(); return p4c_.BoostToCM(); }
    /// magnitude of momentum vector                                                      
    virtual double p() const { if (!unpacked_) unpack(); return p4c_.P(); }
    /// energy                                                                            
    virtual double energy() const { if (!unpacked_) unpack(); return p4c_.E(); }
    /// transverse energy                                                                 
    virtual double et() const { if (!unpacked_) unpack(); return p4_.Et(); }
    /// mass                                                                              
    virtual float mass() const { if (!unpacked_) unpack(); return p4_.M(); }
    /// mass squared                                                                      
    virtual float massSqr() const { if (!unpacked_) unpack(); return p4_.M()*p4_.M(); }

    /// transverse mass                                                                   
    virtual double mt() const { if (!unpacked_) unpack(); return p4_.Mt(); }
    /// transverse mass squared                                                           
    virtual double mtSqr() const { if (!unpacked_) unpack(); return p4_.Mt2(); }
    /// x coordinate of momentum vector                                                   
    virtual double px() const { if (!unpacked_) unpack(); return p4c_.Px(); }
    /// y coordinate of momentum vector                                                   
    virtual double py() const { if (!unpacked_) unpack(); return p4c_.Py(); }
    /// z coordinate of momentum vector                                                   
    virtual double pz() const { if (!unpacked_) unpack(); return p4c_.Pz(); }
    /// transverse momentum                                                               
    virtual float pt() const { if (!unpacked_) unpack(); return p4_.Pt();}
    /// momentum azimuthal angle                                                          
    virtual float phi() const { if (!unpacked_) unpack(); return p4_.Phi(); }
    /// momentum polar angle                                                              
    virtual double theta() const { if (!unpacked_) unpack(); return p4_.Theta(); }
    /// momentum pseudorapidity                                                           
    virtual float eta() const { if (!unpacked_) unpack(); return p4_.Eta(); }
    /// rapidity                                                                          
    virtual double rapidity() const { if (!unpacked_) unpack(); return p4_.Rapidity(); }
    /// rapidity                                                                          
    virtual double y() const { if (!unpacked_) unpack(); return p4_.Rapidity(); }
    /// set 4-momentum                                                                    
    virtual void setP4( const LorentzVector & p4 ) { 
        unpack(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        p4_ = PolarLorentzVector(p4.Pt(), p4.Eta(), p4.Phi(), p4.M());
        pack();
    }
    /// set 4-momentum                                                                    
    virtual void setP4( const PolarLorentzVector & p4 ) { 
        unpack(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        p4_ = p4; 
        pack();
    }
    /// set particle mass                                                                 
    virtual void setMass( double m ) {
      if (!unpacked_) unpack(); 
      p4_ = PolarLorentzVector(p4_.Pt(), p4_.Eta(), p4_.Phi(), m); 
      pack();
    }
    virtual void setPz( double pz ) {
      unpack(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      p4c_ = LorentzVector(p4c_.Px(), p4c_.Py(), pz, p4c_.E());
      p4_  = PolarLorentzVector(p4c_.Pt(), p4c_.Eta(), p4c_.Phi(), p4c_.M());
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

    /* template<typename T> T get() const { */
    /*   if ( hasMasterClone() ) return masterClone()->get<T>(); */
    /*   else return reco::get<T>( * this ); */
    /* } */
    /* /// get a component                                                                                                 */
    /* template<typename T, typename Tag> T get() const { */
    /*   if ( hasMasterClone() ) return masterClone()->get<T, Tag>(); */
    /*   else return reco::get<T, Tag>( * this ); */
    /* } */
    /* /// get a component                                                                                                 */
    /* template<typename T> T get( size_type i ) const { */
    /*   if ( hasMasterClone() ) return masterClone()->get<T>( i ); */
    /*   else return reco::get<T>( * this, i ); */
    /* } */
    /* /// get a component                                                                                                 */
    /* template<typename T, typename Tag> T get( size_type i ) const { */
    /*   if ( hasMasterClone() ) return masterClone()->get<T, Tag>( i ); */
    /*   else return reco::get<T, Tag>( * this, i ); */
    /* } */
    /* /// number of components                                                                                            */
    /* template<typename T> size_type numberOf() const { */
    /*   if ( hasMasterClone() ) return masterClone()->numberOf<T>(); */
    /*   else return reco::numberOf<T>( * this ); */
    /* } */
    /* /// number of components                                                                                            */
    /* template<typename T, typename Tag> size_type numberOf() const { */
    /*   if ( hasMasterClone() ) return masterClone()->numberOf<T, Tag>(); */
    /*   else return reco::numberOf<T, Tag>( * this ); */
    /* } */

    /* template<typename S> */
    /*   struct daughter_iterator   { */
    /*     typedef boost::filter_iterator<S, const_iterator> type; */
    /*   }; */

    /* template<typename S> */
    /*   typename daughter_iterator<S>::type beginFilter( const S & s ) const { */
    /*   return boost::make_filter_iterator(s, begin(), end()); */
    /* } */
    /* template<typename S> */
    /*   typename daughter_iterator<S>::type endFilter( const S & s ) const { */
    /*   return boost::make_filter_iterator(s, end(), end()); */
    /* } */


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
    uint16_t packedPt_, packedEta_, packedPhi_, packedM_;
    void pack(bool unpackAfterwards=true) ;
    void unpack() const ;
 
    /// the four vector                                                 
    mutable PolarLorentzVector p4_;
    mutable LorentzVector p4c_;
    /// vertex position                                                                   
    mutable Point vertex_;
    mutable float dxy_, dz_, dphi_;
    /// PDG identifier                                                                    
    int pdgId_;
    /// Charge
    int8_t charge_;
    ///Ref to first mother
    reco::GenParticleRef mother_;
    // is the momentum p4 unpacked
    mutable bool unpacked_;

    /// check overlap with another Candidate                                              
    virtual bool overlap( const reco::Candidate & ) const;
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  private:
    // const iterator implementation
    typedef reco::candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    // iterator implementation
    typedef reco::candidate::iterator_imp_specific<daughters> iterator_imp_specific;
  };

  typedef std::vector<pat::PackedGenParticle> PackedGenParticleCollection;
  typedef edm::Ref<pat::PackedGenParticleCollection> PackedGenParticleRef;
  typedef edm::RefVector<pat::PackedGenParticleCollection> PackedGenParticleRefVector;
}

#endif
