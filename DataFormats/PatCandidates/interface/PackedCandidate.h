#ifndef __DataFormats_PatCandidates_PackedCandidate_h__
#define __DataFormats_PatCandidates_PackedCandidate_h__

#include <atomic>
#include <mutex>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h" 
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "DataFormats/PatCandidates/interface/CovarianceParameterization.h"

/* #include "DataFormats/Math/interface/PtEtaPhiMass.h" */

//forward declare testing structure
class testPackedCandidate;

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
    PackedCandidate() :
      packedPt_(0), packedEta_(0),
      packedPhi_(0), packedM_(0),
      packedDxy_(0), packedDz_(0), packedDPhi_(0), packedDEta_(0),packedDTrkPt_(0),
      packedCovariance_(),
      packedPuppiweight_(0),
      packedPuppiweightNoLepDiff_(0),
      rawCaloFraction_(0),
      hcalFraction_(0),
      isIsolatedChargedHadron_(0),
      p4_(new PolarLorentzVector(0,0,0,0)), p4c_( new LorentzVector(0,0,0,0)), 
      vertex_(new Point(0,0,0)), dphi_(0), deta_(0), dtrkpt_(0),track_(nullptr), pdgId_(0),
      qualityFlags_(0), pvRefKey_(reco::VertexRef::invalidKey()),
      m_(nullptr), packedHits_(0), packedLayers_(0), normalizedChi2_(0),covarianceVersion_(0),covarianceSchema_(0) { }

    explicit PackedCandidate( const reco::Candidate & c,
                              const reco::VertexRefProd &pvRefProd,
                              reco::VertexRef::key_type pvRefKey) :
      packedPuppiweight_(0), packedPuppiweightNoLepDiff_(0), rawCaloFraction_(0), hcalFraction_(0),
      isIsolatedChargedHadron_(0),
      p4_( new PolarLorentzVector(c.pt(), c.eta(), c.phi(), c.mass())), 
      p4c_( new LorentzVector(*p4_)), vertex_( new Point(c.vertex())), 
      dphi_(0), deta_(0), dtrkpt_(0),
      track_(nullptr), pdgId_(c.pdgId()), qualityFlags_(0), pvRefProd_(pvRefProd),
      pvRefKey_(pvRefKey),m_(nullptr), packedHits_(0), packedLayers_(0),
      normalizedChi2_(0),covarianceVersion_(0), covarianceSchema_(0) {
      packBoth();
    }

    explicit PackedCandidate( const PolarLorentzVector &p4, const Point &vtx,
                              float trkPt,float etaAtVtx,float phiAtVtx,int pdgId,
                              const reco::VertexRefProd &pvRefProd,
                              reco::VertexRef::key_type pvRefKey) :
      packedPuppiweight_(0), packedPuppiweightNoLepDiff_(0), rawCaloFraction_(0), hcalFraction_(0),
      isIsolatedChargedHadron_(0),
      p4_( new PolarLorentzVector(p4) ), p4c_( new LorentzVector(*p4_)), 
      vertex_( new Point(vtx) ), 
      dphi_(reco::deltaPhi(phiAtVtx,p4_.load()->phi())), 
      deta_(std::abs(etaAtVtx-p4_.load()->eta())>=kMinDEtaToStore_ ? etaAtVtx-p4_.load()->eta() : 0.), 
      dtrkpt_(std::abs(trkPt-p4_.load()->pt())>=kMinDTrkPtToStore_ ? trkPt-p4_.load()->pt() : 0.),
      track_(nullptr), pdgId_(pdgId),
      qualityFlags_(0), pvRefProd_(pvRefProd), pvRefKey_(pvRefKey),
      m_(nullptr),packedHits_(0), packedLayers_(0),normalizedChi2_(0),covarianceVersion_(0),covarianceSchema_(0) {
      packBoth();
    }

    explicit PackedCandidate( const LorentzVector &p4, const Point &vtx,
                              float trkPt, float etaAtVtx, float phiAtVtx, int pdgId,
                              const reco::VertexRefProd &pvRefProd,
                              reco::VertexRef::key_type pvRefKey) :
      packedPuppiweight_(0), packedPuppiweightNoLepDiff_(0), rawCaloFraction_(0), hcalFraction_(0),
      isIsolatedChargedHadron_(0),
      p4_(new PolarLorentzVector(p4.Pt(), p4.Eta(), p4.Phi(), p4.M())), 
      p4c_( new LorentzVector(p4)), vertex_( new Point(vtx) ) ,
      dphi_(reco::deltaPhi(phiAtVtx,p4_.load()->phi())), 
      deta_(std::abs(etaAtVtx-p4_.load()->eta())>=kMinDEtaToStore_ ? etaAtVtx-p4_.load()->eta() : 0.), 
      dtrkpt_(std::abs(trkPt-p4_.load()->pt())>=kMinDTrkPtToStore_ ? trkPt-p4_.load()->pt() : 0.),
      track_(nullptr), pdgId_(pdgId), qualityFlags_(0),
      pvRefProd_(pvRefProd), pvRefKey_(pvRefKey),
      m_(nullptr),packedHits_(0),packedLayers_(0),normalizedChi2_(0),covarianceVersion_(0),covarianceSchema_(0)  {
      packBoth();
    }

    PackedCandidate( const PackedCandidate& iOther) :
      packedPt_(iOther.packedPt_), packedEta_(iOther.packedEta_),
      packedPhi_(iOther.packedPhi_), packedM_(iOther.packedM_),
      packedDxy_(iOther.packedDxy_), packedDz_(iOther.packedDz_), 
      packedDPhi_(iOther.packedDPhi_),packedDEta_(iOther.packedDEta_),packedDTrkPt_(iOther.packedDTrkPt_),
      packedCovariance_(iOther.packedCovariance_),
      packedPuppiweight_(iOther.packedPuppiweight_), 
      packedPuppiweightNoLepDiff_(iOther.packedPuppiweightNoLepDiff_),
      rawCaloFraction_(iOther.rawCaloFraction_),
      hcalFraction_(iOther.hcalFraction_),
      isIsolatedChargedHadron_(iOther.isIsolatedChargedHadron_),
      //Need to trigger unpacking in iOther
      p4_( new PolarLorentzVector(iOther.polarP4() ) ),
      p4c_( new LorentzVector(iOther.p4())), vertex_( new Point(iOther.vertex())),
      dxy_(iOther.dxy_), dz_(iOther.dz_),
      dphi_(iOther.dphi_), deta_(iOther.deta_), dtrkpt_(iOther.dtrkpt_),
      track_( iOther.track_ ? new reco::Track(*iOther.track_) : nullptr),
      pdgId_(iOther.pdgId_), qualityFlags_(iOther.qualityFlags_), 
      pvRefProd_(iOther.pvRefProd_),pvRefKey_(iOther.pvRefKey_),
      m_(iOther.m_? new reco::TrackBase::CovarianceMatrix(*iOther.m_) : nullptr),
      packedHits_(iOther.packedHits_),packedLayers_(iOther.packedLayers_),  normalizedChi2_(iOther.normalizedChi2_),
      covarianceVersion_(iOther.covarianceVersion_), covarianceSchema_(iOther.covarianceSchema_)  {
      }

    PackedCandidate( PackedCandidate&& iOther) :
      packedPt_(iOther.packedPt_), packedEta_(iOther.packedEta_),
      packedPhi_(iOther.packedPhi_), packedM_(iOther.packedM_),
      packedDxy_(iOther.packedDxy_), packedDz_(iOther.packedDz_), 
      packedDPhi_(iOther.packedDPhi_), packedDEta_(iOther.packedDEta_), packedDTrkPt_(iOther.packedDTrkPt_),  
      packedCovariance_(iOther.packedCovariance_),
      packedPuppiweight_(iOther.packedPuppiweight_), 
      packedPuppiweightNoLepDiff_(iOther.packedPuppiweightNoLepDiff_),
      rawCaloFraction_(iOther.rawCaloFraction_),
      hcalFraction_(iOther.hcalFraction_),
      isIsolatedChargedHadron_(iOther.isIsolatedChargedHadron_),
      p4_( iOther.p4_.exchange(nullptr) ) ,
      p4c_( iOther.p4c_.exchange(nullptr)), vertex_(iOther.vertex_.exchange(nullptr)),
      dxy_(iOther.dxy_), dz_(iOther.dz_),
      dphi_(iOther.dphi_), deta_(iOther.deta_), dtrkpt_(iOther.dtrkpt_),
      track_( iOther.track_.exchange(nullptr) ),
      pdgId_(iOther.pdgId_), qualityFlags_(iOther.qualityFlags_), 
      pvRefProd_(std::move(iOther.pvRefProd_)),pvRefKey_(iOther.pvRefKey_),
      m_( iOther.m_.exchange(nullptr)),
      packedHits_(iOther.packedHits_), packedLayers_(iOther.packedLayers_),normalizedChi2_(iOther.normalizedChi2_),
      covarianceVersion_(iOther.covarianceVersion_), covarianceSchema_(iOther.covarianceSchema_) {
      }


    PackedCandidate& operator=(const PackedCandidate& iOther) {
      if(this == &iOther) {
        return *this;
      }
      packedPt_ =iOther.packedPt_;
      packedEta_=iOther.packedEta_;
      packedPhi_=iOther.packedPhi_; 
      packedM_=iOther.packedM_;
      packedDxy_=iOther.packedDxy_; 
      packedDz_=iOther.packedDz_; 
      packedDPhi_=iOther.packedDPhi_;
      packedDEta_=iOther.packedDEta_;
      packedDTrkPt_=iOther.packedDTrkPt_;
      packedCovariance_=iOther.packedCovariance_;
      packedPuppiweight_=iOther.packedPuppiweight_; 
      packedPuppiweightNoLepDiff_=iOther.packedPuppiweightNoLepDiff_;
      rawCaloFraction_=iOther.rawCaloFraction_;
      hcalFraction_=iOther.hcalFraction_;
      isIsolatedChargedHadron_=iOther.isIsolatedChargedHadron_;
      //Need to trigger unpacking in iOther
      if(p4_) {
        *p4_ = iOther.polarP4();
      } else {
        p4_.store( new PolarLorentzVector(iOther.polarP4() ) ) ;
      }
      if(p4c_) {
        *p4c_ = iOther.p4();
      } else {
        p4c_.store( new LorentzVector(iOther.p4()));
      }
      if(vertex_) {
        *vertex_ = iOther.vertex();
      } else {
        vertex_.store( new Point(iOther.vertex()));
      }
      dxy_=iOther.dxy_;
      dz_ = iOther.dz_;
      dphi_=iOther.dphi_; 
      deta_=iOther.deta_; 
      dtrkpt_=iOther.dtrkpt_; 
      
      if(!iOther.track_) {
        delete track_.exchange(nullptr);
      } else {
        if(!track_) {
          track_.store( new reco::Track(*iOther.track_));
        } else {
          *track_ = *(iOther.track_);
        }
      }

      pdgId_=iOther.pdgId_; 
      qualityFlags_=iOther.qualityFlags_; 
      pvRefProd_=iOther.pvRefProd_;
      pvRefKey_=iOther.pvRefKey_;
      if(!iOther.m_) {
        delete m_.exchange(nullptr);
       } else {
        if(!m_) {
          m_.store( new reco::Track::CovarianceMatrix(*iOther.m_));
        } else {
          *m_ = *(iOther.m_);
        }
      }
      
	
      packedHits_=iOther.packedHits_; 
      packedLayers_=iOther.packedLayers_; 
      normalizedChi2_=iOther.normalizedChi2_;
      covarianceVersion_=iOther.covarianceVersion_;
      covarianceSchema_=iOther.covarianceSchema_;
      return *this;
    }

    PackedCandidate& operator=(PackedCandidate&& iOther) {
      if(this == &iOther) {
        return *this;
      }
      packedPt_ =iOther.packedPt_;
      packedEta_=iOther.packedEta_;
      packedPhi_=iOther.packedPhi_; 
      packedM_=iOther.packedM_;
      packedDxy_=iOther.packedDxy_; 
      packedDz_=iOther.packedDz_; 
      packedDPhi_=iOther.packedDPhi_;
      packedDEta_=iOther.packedDEta_;
      packedDTrkPt_=iOther.packedDTrkPt_;
      packedCovariance_=iOther.packedCovariance_;
      packedPuppiweight_=iOther.packedPuppiweight_; 
      packedPuppiweightNoLepDiff_=iOther.packedPuppiweightNoLepDiff_;
      rawCaloFraction_=iOther.rawCaloFraction_;
      hcalFraction_=iOther.hcalFraction_;
      isIsolatedChargedHadron_=iOther.isIsolatedChargedHadron_;
      delete p4_.exchange(iOther.p4_.exchange(nullptr));
      delete p4c_.exchange(iOther.p4c_.exchange(nullptr));
      delete vertex_.exchange(iOther.vertex_.exchange(nullptr));
      dxy_=iOther.dxy_;
      dz_ = iOther.dz_;
      dphi_=iOther.dphi_;      
      deta_=iOther.deta_; 
      dtrkpt_=iOther.dtrkpt_;
      delete track_.exchange(iOther.track_.exchange( nullptr));
      pdgId_=iOther.pdgId_; 
      qualityFlags_=iOther.qualityFlags_; 
      pvRefProd_=iOther.pvRefProd_;
      pvRefKey_=iOther.pvRefKey_;
      delete m_.exchange(iOther.m_.exchange(nullptr));
      packedHits_=iOther.packedHits_; 
      packedLayers_=iOther.packedLayers_; 
      normalizedChi2_=iOther.normalizedChi2_;
      covarianceVersion_=iOther.covarianceVersion_;
      covarianceSchema_=iOther.covarianceSchema_;
      return *this;
    }

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
    virtual double massSqr() const { if (!p4c_) unpack(); auto m = p4_.load()->M(); return m*m;}

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
   
    /// pt from the track (normally identical to pt())
    virtual double ptTrk() const {
        maybeUnpackBoth(); 
	return p4_.load()->pt() + dtrkpt_;
    }
    /// momentum azimuthal angle from the track (normally identical to phi())
    virtual float phiAtVtx() const { 
        maybeUnpackBoth(); 
        float ret = p4_.load()->Phi() + dphi_; 
        while (ret >  float(M_PI)) ret -= 2*float(M_PI); 
        while (ret < -float(M_PI)) ret += 2*float(M_PI); 
        return ret; 
    } 
    /// eta from the track (normally identical to eta())
    virtual float etaAtVtx() const { 
        maybeUnpackBoth(); 
        return p4_.load()->eta() + deta_; 
    }
    
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
        maybeUnpackBoth(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        *p4_ = PolarLorentzVector(p4.Pt(), p4.Eta(), p4.Phi(), p4.M());
        packBoth();
    }
    /// set 4-momentum                                                                    
    virtual void setP4( const PolarLorentzVector & p4 ) { 
        maybeUnpackBoth(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
        *p4_ = p4; 
        packBoth();
    }
    /// set particle mass                                                                 
    virtual void setMass( double m ) {
      if (!p4c_) unpack(); 
      *p4_ = PolarLorentzVector(p4_.load()->Pt(), p4_.load()->Eta(), p4_.load()->Phi(), m); 
      pack();
    }
    virtual void setPz( double pz ) {
      maybeUnpackBoth(); // changing px,py,pz changes also mapping between dxy,dz and x,y,z
      *p4c_ = LorentzVector(p4c_.load()->Px(), p4c_.load()->Py(), pz, p4c_.load()->E());
      *p4_  = PolarLorentzVector(p4c_.load()->Pt(), p4c_.load()->Eta(), p4c_.load()->Phi(), p4c_.load()->M());
      packBoth();
    }
    /// set impact parameters covariance

    // Note: mask is also the maximum value
    enum trackHitShiftsAndMasks {
      trackPixelHitsMask = 7,
      trackStripHitsMask = 31, trackStripHitsShift = 3
    };

    // set number of tracker hits and layers
    virtual void setHits( const reco::Track & tk) {
      // first we count the number of layers with hits
      int numberOfPixelLayers_ = tk.hitPattern().pixelLayersWithMeasurement();
      if (numberOfPixelLayers_ > trackPixelHitsMask) numberOfPixelLayers_ = trackPixelHitsMask;
      int numberOfStripLayers_ = tk.hitPattern().stripLayersWithMeasurement();
      if (numberOfStripLayers_ > trackStripHitsMask) numberOfStripLayers_ = trackStripHitsMask;
      packedLayers_ = (numberOfPixelLayers_&trackPixelHitsMask) | (numberOfStripLayers_ << trackStripHitsShift);
      // now we count number of additional hits, beyond the one-per-layer implied by packedLayers_
      int numberOfPixelHits_ = tk.hitPattern().numberOfValidPixelHits() - numberOfPixelLayers_;
      if (numberOfPixelHits_ > trackPixelHitsMask) numberOfPixelHits_ = trackPixelHitsMask;
      int numberOfStripHits_ = tk.hitPattern().numberOfValidHits() - numberOfPixelHits_ - numberOfPixelLayers_ - numberOfStripLayers_;
      if (numberOfStripHits_ > trackStripHitsMask) numberOfStripHits_ = trackStripHitsMask;

      packedHits_ = (numberOfPixelHits_&trackPixelHitsMask) | (numberOfStripHits_ << trackStripHitsShift);
    }
  
    virtual void setTrackProperties( const reco::Track & tk, const reco::Track::CovarianceMatrix & covariance,int quality,int covarianceVersion) {
      covarianceVersion_ = covarianceVersion ;
      covarianceSchema_ = quality ;
      normalizedChi2_ = tk.normalizedChi2();
      setHits(tk);
      packBoth();
      packCovariance(covariance,false); 
    }

    // set track properties using quality and covarianceVersion to define the level of details in the cov. matrix	
    virtual void setTrackProperties( const reco::Track & tk,int quality,int covarianceVersion ) {
	setTrackProperties(tk,tk.covariance(), quality,covarianceVersion );
    }	
 
    int numberOfPixelHits() const { return (packedHits_ & trackPixelHitsMask) + pixelLayersWithMeasurement(); }
    int numberOfHits() const { return (packedHits_ >> trackStripHitsShift) + stripLayersWithMeasurement() + numberOfPixelHits(); }
    int pixelLayersWithMeasurement() const { return packedLayers_ & trackPixelHitsMask; }
    int stripLayersWithMeasurement() const { return (packedLayers_ >> trackStripHitsShift); }
    int trackerLayersWithMeasurement() const { return stripLayersWithMeasurement() + pixelLayersWithMeasurement(); }
    virtual void setCovarianceVersion(int v) {covarianceVersion_=v;}
    int covarianceVersion() const { return covarianceVersion_;}
    int covarianceSchema() const { return covarianceSchema_;}

	
    /// vertex position
    virtual const Point & vertex() const { maybeUnpackBoth(); return *vertex_; }//{ if (fromPV_) return Point(0,0,0); else return Point(0,0,100); }
    /// x coordinate of vertex position                                                   
    virtual double vx() const  { maybeUnpackBoth(); return vertex_.load()->X(); }//{ return 0; }
    /// y coordinate of vertex position                                                   
    virtual double vy() const  { maybeUnpackBoth(); return vertex_.load()->Y(); }//{ return 0; }
    /// z coordinate of vertex position                                                   
    virtual double vz() const  { maybeUnpackBoth(); return vertex_.load()->Z(); }//{ if (fromPV_) return 0; else return 100; }
    /// set vertex                                                                        
    virtual void setVertex( const Point & vertex ) { maybeUnpackBoth(); *vertex_ = vertex; packVtx(); }

    ///This refers to the association to PV=ipv. >=PVLoose corresponds to JME definition, >=PVTight to isolation definition
    enum PVAssoc { NoPV=0, PVLoose=1, PVTight=2, PVUsedInFit=3 } ;
    const PVAssoc fromPV(size_t ipv=0) const {
        reco::VertexRef pvRef = vertexRef();
        if(pvAssociationQuality()==UsedInFitTight and pvRef.key()==ipv) return PVUsedInFit;
        if(pvRef.key()==ipv or abs(pdgId())==13 or abs(pdgId())==11 ) return PVTight;
        if(pvAssociationQuality() == CompatibilityBTag and std::abs(dzAssociatedPV()) >  std::abs(dz(ipv))) return PVTight; // it is not closest, but at least prevents the B assignment stealing
        if(pvAssociationQuality() < UsedInFitLoose or pvRef->ndof() < 4.0 ) return PVLoose;
        return NoPV;
    }

    /// The following contains information about how the association to the PV, given in vertexRef, is obtained.
    ///
    enum PVAssociationQuality { NotReconstructedPrimary=0,OtherDeltaZ=1,CompatibilityBTag=4,CompatibilityDz=5,UsedInFitLoose=6,UsedInFitTight=7};
    const PVAssociationQuality pvAssociationQuality() const { return PVAssociationQuality((qualityFlags_ & assignmentQualityMask)>>assignmentQualityShift); }
    void setAssociationQuality( PVAssociationQuality q )   {  qualityFlags_ = (qualityFlags_ & ~assignmentQualityMask) | ((q << assignmentQualityShift) & assignmentQualityMask);  }

    const reco::VertexRef vertexRef() const { return reco::VertexRef(pvRefProd_, pvRefKey_); }

    /// dxy with respect to the PV ref
    virtual float dxy() const { maybeUnpackBoth(); return dxy_; }
    /// dz with respect to the PV[ipv]
    virtual float dz(size_t ipv=0)  const { maybeUnpackBoth(); return dz_ + (*pvRefProd_)[pvRefKey_].position().z()-(*pvRefProd_)[ipv].position().z(); }
    /// dz with respect to the PV ref
    virtual float dzAssociatedPV()  const { maybeUnpackBoth(); return dz_; }
    /// dxy with respect to another point
    virtual float dxy(const Point &p) const ;
    /// dz  with respect to another point
    virtual float dz(const Point &p)  const ;

    /// uncertainty on dz 
    virtual float dzError() const { maybeUnpackCovariance(); return sqrt((*m_.load())(4,4)); }
    /// uncertainty on dxy
    virtual float dxyError() const { maybeUnpackCovariance(); return sqrt((*m_.load())(3,3)); }


    /// Return reference to a pseudo track made with candidate kinematics, parameterized error for eta,phi,pt and full IP covariance	
    virtual const reco::Track & pseudoTrack() const { if (!track_) unpackTrk(); return *track_; }

    /// return a pointer to the track if present. otherwise, return a null pointer
    virtual const reco::Track * bestTrack() const {
      if (packedHits_!=0 || packedLayers_ !=0) {
        maybeUnpackTrack();
        return track_.load();
      }
      else
        return nullptr;
    }
    /// Return true if a bestTrack can be extracted from this Candidate
    bool hasTrackDetails() const {return  (packedHits_!=0 || packedLayers_ !=0); }
      

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

    // puppiweights
    void setPuppiWeight(float p, float p_nolep = 0.0);  /// Set both weights at once (with option for only full PUPPI)
    float puppiWeight() const;                          /// Weight from full PUPPI
    float puppiWeightNoLep() const;                     /// Weight from PUPPI removing leptons
    
    // for the neutral fractions
    void setRawCaloFraction(float p);                      /// Set the raw ECAL+HCAL energy over candidate energy for isolated charged hadrons
    float rawCaloFraction() const { return (rawCaloFraction_/100.); }    /// Raw ECAL+HCAL energy over candidate energy for isolated charged hadrons
    void setHcalFraction(float p);                      /// Set the fraction of Hcal needed for HF and neutral hadrons and isolated charged hadrons
    float hcalFraction() const { return (hcalFraction_/100.); }    /// Fraction of Ecal and Hcal for HF and neutral hadrons and isolated charged hadrons

     // isolated charged hadrons
    void setIsIsolatedChargedHadron(bool p);                      /// Set isolation (as in particle flow, i.e. at calorimeter surface rather than at PV) flat for charged hadrons
    bool isIsolatedChargedHadron() const { return isIsolatedChargedHadron_; }    /// Flag isolation (as in particle flow, i.e. at calorimeter surface rather than at PV) flag for charged hadrons

    struct PackedCovariance {
        PackedCovariance(): dxydxy(0),dxydz(0),dzdz(0),dlambdadz(0),dphidxy(0),dptdpt(0),detadeta(0),dphidphi(0) {}
        //3D IP covariance
        uint16_t dxydxy;
        uint16_t dxydz;
        uint16_t dzdz;
        //other IP relevant elements
        uint16_t dlambdadz;
        uint16_t dphidxy;
        //other diag elements
        uint16_t dptdpt;
        uint16_t detadeta;
        uint16_t dphidphi;
    };

  private:
    void unpackCovarianceElement(reco::TrackBase::CovarianceMatrix & m, uint16_t packed, int i,int j) const {
      m(i,j)= covarianceParameterization().unpack(packed,covarianceSchema_,i,j,pt(),eta(),numberOfHits(), numberOfPixelHits());
    }
    uint16_t packCovarianceElement(const reco::TrackBase::CovarianceMatrix &m,int i,int j) const {
      return covarianceParameterization().pack(m(i,j),covarianceSchema_,i,j,pt(),eta(),numberOfHits(), numberOfPixelHits());
    }

  protected:
    friend class ::testPackedCandidate;
    static constexpr float kMinDEtaToStore_=0.001;
    static constexpr float kMinDTrkPtToStore_=0.001;
    

    uint16_t packedPt_, packedEta_, packedPhi_, packedM_;
    uint16_t packedDxy_, packedDz_, packedDPhi_, packedDEta_, packedDTrkPt_;
    PackedCovariance packedCovariance_;

    void pack(bool unpackAfterwards=true) ;
    void unpack() const ;
    void packVtx(bool unpackAfterwards=true) ;
    void unpackVtx() const ;
    void packCovariance(const reco::TrackBase::CovarianceMatrix  & m,bool unpackAfterwards=true) ;
    void unpackCovariance() const;
    void maybeUnpackBoth() const { if (!p4c_) unpack(); if (!vertex_) unpackVtx(); }
    void maybeUnpackTrack() const { if (!track_) unpackTrk(); }
    void maybeUnpackCovariance() const { if (!m_) unpackCovariance(); }
    void packBoth() { pack(false); packVtx(false); delete p4_.exchange(nullptr); delete p4c_.exchange(nullptr); delete vertex_.exchange(nullptr); unpack(); unpackVtx(); } // do it this way, so that we don't loose precision on the angles before computing dxy,dz
    void unpackTrk() const ;

    int8_t packedPuppiweight_;
    int8_t packedPuppiweightNoLepDiff_; // storing the DIFFERENCE of (all - "no lep") for compression optimization
    uint8_t rawCaloFraction_;
    int8_t hcalFraction_;

    bool isIsolatedChargedHadron_;

    /// the four vector                                                 
    mutable std::atomic<PolarLorentzVector*> p4_;
    mutable std::atomic<LorentzVector*> p4c_;
    /// vertex position                                                                   
    mutable std::atomic<Point*> vertex_;
    CMS_THREAD_GUARD(vertex_) mutable float dxy_, dz_, dphi_, deta_, dtrkpt_;
    /// reco::Track                                                                   
    mutable std::atomic<reco::Track*> track_;
    /// PDG identifier                                                                    
    int pdgId_;
    uint16_t qualityFlags_;
    /// Use these to build a Ref to primary vertex
    reco::VertexRefProd pvRefProd_;
    reco::VertexRef::key_type pvRefKey_;

    /// IP covariance	
    mutable std::atomic<reco::TrackBase::CovarianceMatrix *> m_;
    uint8_t packedHits_, packedLayers_; // packedLayers_ -> layers with valid hits; packedHits_ -> extra hits beyond the one-per-layer implied by packedLayers_
    
    /// track quality information
    uint8_t normalizedChi2_; 
    uint16_t covarianceVersion_;
    uint16_t covarianceSchema_;
    CMS_THREAD_SAFE static CovarianceParameterization covarianceParameterization_;
    //static std::atomic<CovarianceParameterization*> covarianceParameterization_;
    static std::once_flag covariance_load_flag;
    const CovarianceParameterization & covarianceParameterization() const {
	std::call_once(covariance_load_flag,[](int v) { covarianceParameterization_.load(v); } ,covarianceVersion_ );
        if(covarianceParameterization_.loadedVersion() != covarianceVersion_ )
        {
          throw edm::Exception(edm::errors::UnimplementedFeature)
           << "Attempting to load multiple covariance version in same process. This is not supported.";
        }
        return  covarianceParameterization_;
    }

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
