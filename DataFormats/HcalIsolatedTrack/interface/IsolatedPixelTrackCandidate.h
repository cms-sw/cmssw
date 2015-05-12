#ifndef HcalIsolatedTrack_IsolatedPixelTrackCandidate_h
#define HcalIsolatedTrack_IsolatedPixelTrackCandidate_h
/** \class reco::IsolatedPixelTrackCandidate
 *
 *
 */

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include <vector>
#include <map>
#include <utility>

namespace reco {
  
  class IsolatedPixelTrackCandidate: public RecoCandidate {

  public:
    
    // default constructor
    IsolatedPixelTrackCandidate() : RecoCandidate() { 
      enIn_=-1;
      enOut_=-1;
      nhitIn_=-1;
      nhitOut_=-1;
      maxPtPxl_=-1;
      sumPtPxl_=-1;
      etaEcal_=0;
      phiEcal_=0;
      etaPhiEcal_=false; 
    }
    ///constructor from LorentzVector
    IsolatedPixelTrackCandidate(const LorentzVector& v): RecoCandidate(0,v) {
      enIn_=-1;
      enOut_=-1;
      nhitIn_=-1;
      nhitOut_=-1;
      maxPtPxl_=-1;
      sumPtPxl_=-1;
      etaEcal_=0;
      phiEcal_=0;
      etaPhiEcal_=false;
    }
    /// constructor from a track
  IsolatedPixelTrackCandidate(const reco::TrackRef & tr, const l1extra::L1JetParticleRef & tauRef, double max, double sum): 
    RecoCandidate( 0, LorentzVector((tr.get()->px()),(tr.get())->py(),(tr.get())->pz(),(tr.get())->p()) ),
      track_(tr), l1tauJet_(tauRef), maxPtPxl_(max), sumPtPxl_(sum) {
      enIn_=-1;
      enOut_=-1;
      nhitIn_=-1;
      nhitOut_=-1;
      etaEcal_=0;
      phiEcal_=0;
      etaPhiEcal_=false;
    }
        
    ///constructor from tau jet
    IsolatedPixelTrackCandidate(const l1extra::L1JetParticleRef & tauRef, double enIn, double enOut, int nhitIn, int nhitOut):
    RecoCandidate( 0, LorentzVector(tauRef->px(),tauRef->py(),tauRef->pz(),tauRef->p()) ), 
      l1tauJet_(tauRef), enIn_(enIn), enOut_(enOut), nhitIn_(nhitIn), nhitOut_(nhitOut) {
      maxPtPxl_=-1;
      sumPtPxl_=-1;
      etaEcal_=0;
      phiEcal_=0;
      etaPhiEcal_=false;
    }
    /// Copy constructor
    IsolatedPixelTrackCandidate(const IsolatedPixelTrackCandidate&);

    /// destructor
    virtual ~IsolatedPixelTrackCandidate();

    /// returns a clone of the candidate
    virtual IsolatedPixelTrackCandidate * clone() const;

    /// refrence to a Track
    virtual reco::TrackRef track() const;
    void setTrack( const reco::TrackRef & tr ) { track_ = tr; }

    /// highest Pt of other pixel tracks in the cone around the candidate
    double maxPtPxl() const {return maxPtPxl_;}
    void   setMaxPtPxl(double mptpxl) {maxPtPxl_=mptpxl;}

    /// Pt sum of other pixel tracks in the cone around the candidate
    double sumPtPxl() const {return sumPtPxl_;}
    void   setSumPtPxl(double sumptpxl) {sumPtPxl_=sumptpxl;}
          
    /// get reference to L1 tau jet
    virtual l1extra::L1JetParticleRef l1tau() const;
    void    setL1TauJet( const l1extra::L1JetParticleRef & tauRef ) { l1tauJet_ = tauRef; }
          
    /// ECAL energy in the inner cone around tau jet
    double energyIn() const {return enIn_; }
    void   setEnergyIn(double a) {enIn_=a;}
          
    /// ECAL energy in the outer cone around tau jet
    double energyOut() const {return enOut_;}
    void   setEnergyOut(double a) {enOut_=a;}
          
    /// number of ECAL hits in the inner cone around tau jet
    int    nHitIn() const {return nhitIn_;}
    void   setNHitIn(int a) {nhitIn_=a;}
          
    /// number of ECAL hits in the outer cone around tau jet
    int    nHitOut() const {return nhitOut_;}
    void   setNHitOut(int a) {nhitOut_=a;}
          
    ///get index of tower which track is hitting
    std::pair<int,int> towerIndex() const;

    ///eta, phi at ECAL surface
    void setEtaPhiEcal(double eta, double phi) {
      etaEcal_=eta; phiEcal_=phi; etaPhiEcal_=true;
    }
    std::pair<double,double> etaPhiEcal() const {
      return ((etaPhiEcal_) ? std::pair<double,double>(etaEcal_,phiEcal_) : std::pair<double,double>(0,0));
    }
    bool etaPhiEcalValid() const {return etaPhiEcal_;}

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a Track
    reco::TrackRef track_;
    /// reference to a L1 tau jet
    l1extra::L1JetParticleRef l1tauJet_;
    /// highest Pt of other pixel tracks in the cone around the candidate
    double maxPtPxl_;
    /// Pt sum of other pixel tracks in the cone around the candidate
    double sumPtPxl_;
    /// energy in inner cone around L1 tau jet
    double enIn_;
    /// energy in outer cone around L1 tau jet
    double enOut_;
    /// number of hits in inner cone
    int nhitIn_;
    /// number of hits in inner cone
    int nhitOut_;
    /// eta, phi at ECAL
    bool etaPhiEcal_;
    double etaEcal_, phiEcal_;
  };
  
  
}

#endif
