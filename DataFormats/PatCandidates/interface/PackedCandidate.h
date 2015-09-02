#ifndef __DataFormats_PatCandidates_PackedCandidate_h__
#define __DataFormats_PatCandidates_PackedCandidate_h__

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h" 
/* #include "DataFormats/Math/interface/PtEtaPhiMass.h" */

namespace pat {
  class PackedCandidate : public reco::Candidate {
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
  PackedCandidate()
    : p4_(0,0,0,0), p4c_(0,0,0,0), vertex_(0,0,0), dphi_(0), pdgId_(0), qualityFlags_(0), unpacked_(false), unpackedVtx_(true), unpackedTrk_(false), dxydxy_(0),dzdz_(0),dxydz_(0),dlambdadz_(0),dphidxy_(0),dptdpt_(0),detadeta_(0),dphidphi_(0),packedHits_(0),normalizedChi2_(0) { }
  explicit PackedCandidate( const reco::Candidate & c, const reco::VertexRef &pv)
    : p4_(c.pt(), c.eta(), c.phi(), c.mass()), p4c_(p4_), vertex_(c.vertex()), dphi_(0), pdgId_(c.pdgId()), qualityFlags_(0), pvRef_(pv), unpacked_(true) , unpackedVtx_(true), unpackedTrk_(false), dxydxy_(0),dzdz_(0),dxydz_(0),dlambdadz_(0),dphidxy_(0),dptdpt_(0),detadeta_(0),dphidphi_(0),packedHits_(0),normalizedChi2_(0) { packBoth(); }

  explicit PackedCandidate( const PolarLorentzVector &p4, const Point &vtx, float phiAtVtx, int pdgId, const reco::VertexRef &pv)
    : p4_(p4), p4c_(p4_), vertex_(vtx), dphi_(reco::deltaPhi(phiAtVtx,p4_.phi())), pdgId_(pdgId), qualityFlags_(0), pvRef_(pv), unpacked_(true), unpackedVtx_(true), unpackedTrk_(false),dxydxy_(0),dzdz_(0),dxydz_(0),dlambdadz_(0),dphidxy_(0),dptdpt_(0),detadeta_(0),dphidphi_(0),packedHits_(0),normalizedChi2_(0) { packBoth(); }

  explicit PackedCandidate( const LorentzVector &p4, const Point &vtx, float phiAtVtx, int pdgId, const reco::VertexRef &pv)
    : p4_(p4.Pt(), p4.Eta(), p4.Phi(), p4.M()), p4c_(p4), vertex_(vtx), dphi_(reco::deltaPhi(phiAtVtx,p4_.phi())), pdgId_(pdgId), qualityFlags_(0), pvRef_(pv), unpacked_(true), unpackedVtx_(true), unpackedTrk_(false),dxydxy_(0),dzdz_(0),dxydz_(0),dlambdadz_(0),dphidxy_(0),dptdpt_(0),detadeta_(0),dphidphi_(0),packedHits_(0),normalizedChi2_(0) { packBoth(); }
 
 
    
    
    /// destructor
    virtual ~PackedCandidate();
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
      switch (abs(pdgId_)) {
      case 211: return (pdgId_>0)-(pdgId_<0);
      case 11:  return (-1)*(pdgId_>0)+(pdgId_<0); //e
      case 13:  return (-1)*(pdgId_>0)+(pdgId_<0); //mu
      case 15:  return (-1)*(pdgId_>0)+(pdgId_<0); //tau
      case 24:  return (pdgId_>0)-(pdgId_<0); //W
      default:  return 0;  //FIXME: charge is not defined
      }
    }
    /// set electric charge                                                               
    virtual void setCharge( int charge) {}
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
    double et() const { return (pt()<=0) ? 0 : p4c_.Et(); }  
    /// transverse energy squared (use this for cuts)!
    double et2() const { return (pt()<=0) ? 0 : p4c_.Et2(); }   
    /// mass                                                                              
    virtual double mass() const { if (!unpacked_) unpack(); return p4_.M(); }
    /// mass squared                                                                      
    virtual double massSqr() const { if (!unpacked_) unpack(); return p4_.M()*p4_.M(); }

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
    virtual double pt() const { if (!unpacked_) unpack(); return p4_.Pt();}
    /// momentum azimuthal angle                                                          
    virtual double phi() const { if (!unpacked_) unpack(); return p4_.Phi(); }
    /// momentum azimuthal angle from the track (normally identical to phi())
    virtual float phiAtVtx() const { 
        maybeUnpackBoth(); 
        float ret = p4_.Phi() + dphi_; 
        while (ret >  float(M_PI)) ret -= 2*float(M_PI); 
        while (ret < -float(M_PI)) ret += 2*float(M_PI); 
        return ret; 
    }
    /// momentum polar angle                                                              
    virtual double theta() const { if (!unpacked_) unpack(); return p4_.Theta(); }
    /// momentum pseudorapidity                                                           
    virtual double eta() const { if (!unpacked_) unpack(); return p4_.Eta(); }
    /// rapidity                                                                          
    virtual double rapidity() const { if (!unpacked_) unpack(); return p4_.Rapidity(); }
    /// rapidity                                                                          
    virtual double y() const { if (!unpacked_) unpack(); return p4_.Rapidity(); }
    /// set 4-momentum                                                                    
    virtual void setP4( const LorentzVector & p4 ) { 
        maybeUnpackBoth(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        p4_ = PolarLorentzVector(p4.Pt(), p4.Eta(), p4.Phi(), p4.M());
        packBoth();
    }
    /// set 4-momentum                                                                    
    virtual void setP4( const PolarLorentzVector & p4 ) { 
        maybeUnpackBoth(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        p4_ = p4; 
        packBoth();
    }
    /// set particle mass                                                                 
    virtual void setMass( double m ) {
      if (!unpacked_) unpack(); 
      p4_ = PolarLorentzVector(p4_.Pt(), p4_.Eta(), p4_.Phi(), m); 
      pack();
    }
    virtual void setPz( double pz ) {
      maybeUnpackBoth(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      p4c_ = LorentzVector(p4c_.Px(), p4c_.Py(), pz, p4c_.E());
      p4_  = PolarLorentzVector(p4c_.Pt(), p4c_.Eta(), p4c_.Phi(), p4c_.M());
      packBoth();
    }
    /// set impact parameters covariance

    virtual void setTrackProperties( const reco::Track & tk, const reco::Track::CovarianceMatrix & covariance) {
      dxydxy_ = covariance(3,3);
      dxydz_ = covariance(3,4);
      dzdz_ = covariance(4,4);
      dphidxy_ = covariance(2,3);
      dlambdadz_ = covariance(1,4);
      dptdpt_ = covariance(0,0)*pt()*pt();
      detadeta_ = covariance(1,1);
      dphidphi_ = covariance(2,2)*pt()*pt();

      normalizedChi2_ = tk.normalizedChi2();
      int numberOfPixelHits_ = tk.hitPattern().numberOfValidPixelHits();
      if (numberOfPixelHits_ > 7) numberOfPixelHits_ = 7;
      int numberOfStripHits_ = tk.hitPattern().numberOfValidHits() - numberOfPixelHits_;
      if (numberOfStripHits_ > 31) numberOfStripHits_ = 31;
      packedHits_ = (numberOfPixelHits_&0x7) | (numberOfStripHits_ << 3);
      packBoth();
    }

    virtual void setTrackProperties( const reco::Track & tk ) {
	setTrackProperties(tk,tk.covariance());
    }	
 
    int numberOfPixelHits() const { return packedHits_ & 0x7; }
    int numberOfHits() const { return (packedHits_ >> 3) + numberOfPixelHits(); }
	
    /// vertex position
    virtual const Point & vertex() const { maybeUnpackBoth(); return vertex_; }//{ if (fromPV_) return Point(0,0,0); else return Point(0,0,100); }
    /// x coordinate of vertex position                                                   
    virtual double vx() const  { maybeUnpackBoth(); return vertex_.X(); }//{ return 0; }
    /// y coordinate of vertex position                                                   
    virtual double vy() const  { maybeUnpackBoth(); return vertex_.Y(); }//{ return 0; }
    /// z coordinate of vertex position                                                   
    virtual double vz() const  { maybeUnpackBoth(); return vertex_.Z(); }//{ if (fromPV_) return 0; else return 100; }
    /// set vertex                                                                        
    virtual void setVertex( const Point & vertex ) { maybeUnpackBoth(); vertex_ = vertex; packVtx(); }

    ///This refers to the association to PV=ipv. >=PVLoose corresponds to JME definition, >=PVTight to isolation definition
    enum PVAssoc { NoPV=0, PVLoose=1, PVTight=2, PVUsedInFit=3 } ;
    const PVAssoc fromPV(size_t ipv=0) const { 
        if(pvAssociationQuality()==UsedInFitTight and pvRef_.key()==ipv) return PVUsedInFit;
        if(pvRef_.key()==ipv or abs(pdgId())==13 or abs(pdgId())==11 ) return PVTight;
        if(pvAssociationQuality() == CompatibilityBTag and std::abs(dzAssociatedPV()) >  std::abs(dz(ipv))) return PVTight; // it is not closest, but at least prevents the B assignment stealing
        if(pvAssociationQuality() < UsedInFitLoose or pvRef_->ndof() < 4.0 ) return PVLoose;
        return NoPV;
    }

    /// The following contains information about how the association to the PV, given in vertexRef, is obtained.
    ///
    enum PVAssociationQuality { NotReconstructedPrimary=0,OtherDeltaZ=1,CompatibilityBTag=4,CompatibilityDz=5,UsedInFitLoose=6,UsedInFitTight=7};
    const PVAssociationQuality pvAssociationQuality() const { return PVAssociationQuality((qualityFlags_ & assignmentQualityMask)>>assignmentQualityShift); }
    void setAssociationQuality( PVAssociationQuality q )   {  qualityFlags_ = (qualityFlags_ & ~assignmentQualityMask) | ((q << assignmentQualityShift) & assignmentQualityMask);  }

    /// set reference to the primary vertex                                                                        
    void setVertexRef( const reco::VertexRef & vertexRef ) { maybeUnpackBoth(); pvRef_ = vertexRef; packVtx(); }
    const reco::VertexRef vertexRef() const { return pvRef_; }

    /// dxy with respect to the PV ref
    virtual float dxy() const { maybeUnpackBoth(); return dxy_; }
    /// dz with respect to the PV[ipv]
    virtual float dz(size_t ipv=0)  const { maybeUnpackBoth(); return dz_+pvRef_->position().z()-(*pvRef_.product())[ipv].position().z(); }
    /// dz with respect to the PV ref
    virtual float dzAssociatedPV()  const { maybeUnpackBoth(); return dz_; }
    /// dxy with respect to another point
    virtual float dxy(const Point &p) const ;
    /// dz  with respect to another point
    virtual float dz(const Point &p)  const ;

    /// uncertainty on dz 
    virtual float dzError() const { maybeUnpackBoth(); return sqrt(dzdz_); }
    /// uncertainty on dxy
    virtual float dxyError() const { maybeUnpackBoth(); return sqrt(dxydxy_); }


    /// Return reference to a pseudo track made with candidate kinematics, parameterized error for eta,phi,pt and full IP covariance	
    virtual const reco::Track & pseudoTrack() const { if (!unpackedTrk_) unpackTrk(); return track_; }

    /// return a pointer to the track if present. otherwise, return a null pointer
    virtual const reco::Track * bestTrack() const {
      if (packedHits_!=0) {
        if (!unpackedTrk_) unpackTrk();
        return &track_;
      }
      else
        return nullptr;
    }

    /// true if the track had the highPurity quality bit
    bool trackHighPurity() const { return (qualityFlags_ & trackHighPurityMask)>>trackHighPurityShift; }
    /// set to true if the track had the highPurity quality bit
    void setTrackHighPurity(bool highPurity) {  qualityFlags_ = (qualityFlags_ & ~trackHighPurityMask) | ((highPurity << trackHighPurityShift) & trackHighPurityMask);  }

    /// Enumerator specifying the 
    enum LostInnerHits {
        validHitInFirstPixelBarrelLayer=-1,
        noLostInnerHits=0,   // it could still not have a hit in the first layer, e.g. if it crosses an inactive sensor
        oneLostInnerHit=1,   
        moreLostInnerHits=2
    };
    LostInnerHits lostInnerHits() const {
         return LostInnerHits(int16_t((qualityFlags_ & lostInnerHitsMask)>>lostInnerHitsShift)-1);
    }
    void setLostInnerHits(LostInnerHits hits) {
        int lost = hits; if (lost > 2) lost = 2; // protection against misuse
        lost++; // shift so it's 0 .. 3 instead of (-1) .. 2
        qualityFlags_ = (qualityFlags_ & ~lostInnerHitsMask) | ((lost << lostInnerHitsShift) & lostInnerHitsMask); 
    }

    void setMuonID(bool isStandAlone, bool isGlobal) {
        int16_t muonFlags = isStandAlone | (2*isGlobal);
        qualityFlags_ = (qualityFlags_ & ~muonFlagsMask) | ((muonFlags << muonFlagsShift) & muonFlagsMask);
    }

    /// PDG identifier                                                                    
    virtual int pdgId() const   { return pdgId_; }
    // set PDG identifier                                                                 
    virtual void setPdgId( int pdgId )   { pdgId_ = pdgId; }
    /// status word                                                                       
    virtual int status() const   { return qualityFlags_; } /*FIXME*/
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
    virtual PackedCandidate * clone() const  {
      return new PackedCandidate( *this );
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

    virtual bool isElectron() const { return false; }
    virtual bool isMuon() const { return false; }
    virtual bool isStandAloneMuon() const { return ((qualityFlags_ & muonFlagsMask) >> muonFlagsShift) & 1; }
    virtual bool isGlobalMuon() const { return ((qualityFlags_ & muonFlagsMask) >> muonFlagsShift) & 2; }
    virtual bool isTrackerMuon() const { return false; }
    virtual bool isCaloMuon() const { return false; }
    virtual bool isPhoton() const { return false; }
    virtual bool isConvertedPhoton() const { return false; }
    virtual bool isJet() const { return false; }

    // puppiweight
    void setPuppiWeight(float p, float p_nolep = 0.0);  /// Set both weights at once (with option for only full PUPPI)
    float puppiWeight() const;                          /// Weight from full PUPPI
    float puppiWeightNoLep() const;                     /// Weight from PUPPI removing leptons
    
  protected:
    uint16_t packedPt_, packedEta_, packedPhi_, packedM_;
    uint16_t packedDxy_, packedDz_, packedDPhi_;
    uint16_t packedCovarianceDxyDxy_,packedCovarianceDxyDz_,packedCovarianceDzDz_;
    int8_t packedCovarianceDlambdaDz_,packedCovarianceDphiDxy_;
    int8_t packedCovarianceDptDpt_,packedCovarianceDetaDeta_,packedCovarianceDphiDphi_;
    void pack(bool unpackAfterwards=true) ;
    void unpack() const ;
    void packVtx(bool unpackAfterwards=true) ;
    void unpackVtx() const ;
    void maybeUnpackBoth() const { if (!unpacked_) unpack(); if (!unpackedVtx_) unpackVtx(); }
    void packBoth() { pack(false); packVtx(false); unpack(); unpackVtx(); } // do it this way, so that we don't loose precision on the angles before computing dxy,dz
    void unpackTrk() const ;

    int8_t packedPuppiweight_;
    int8_t packedPuppiweightNoLepDiff_; // storing the DIFFERENCE of (all - "no lep") for compression optimization
    /// the four vector                                                 
    mutable PolarLorentzVector p4_;
    mutable LorentzVector p4c_;
    /// vertex position                                                                   
    mutable Point vertex_;
    mutable float dxy_, dz_, dphi_;
    /// reco::Track                                                                   
    mutable reco::Track track_;
    /// PDG identifier                                                                    
    int pdgId_;
    uint16_t qualityFlags_;
    /// Ref to primary vertex
    edm::Ref<reco::VertexCollection> pvRef_;
    // is the momentum p4 unpacked
    mutable bool unpacked_;
    // are the dxy, dz and vertex unpacked
    mutable bool unpackedVtx_;
    // is the track unpacked
    mutable bool unpackedTrk_;
    /// IP covariance	
    mutable float dxydxy_, dzdz_, dxydz_,dlambdadz_,dphidxy_,dptdpt_,detadeta_,dphidphi_;
    uint8_t packedHits_;
    /// track quality information
    uint8_t normalizedChi2_; 
//    uint8_t numberOfPixelHits_;
  //  uint8_t numberOfHits_;
    
    /// check overlap with another Candidate                                              
    virtual bool overlap( const reco::Candidate & ) const;
    template<typename, typename, typename> friend struct component;
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

    enum qualityFlagsShiftsAndMasks {
        assignmentQualityMask = 0x7, assignmentQualityShift = 0,
        trackHighPurityMask  = 0x8, trackHighPurityShift=3,
        lostInnerHitsMask = 0x30, lostInnerHitsShift=4,
        muonFlagsMask = 0x0600, muonFlagsShift=9
    };
  };

  typedef std::vector<pat::PackedCandidate> PackedCandidateCollection;
  typedef edm::Ref<pat::PackedCandidateCollection> PackedCandidateRef;
  typedef edm::RefVector<pat::PackedCandidateCollection> PackedCandidateRefVector;
}

#endif
