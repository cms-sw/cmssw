#ifndef HcalIsolatedTrack_HcalIsolatedTrackCandidate_h
#define HcalIsolatedTrack_HcalIsolatedTrackCandidate_h
/** \class reco::HcalIsolatedTrackCandidate
 *
 *
 */

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HcalIsolatedTrack/interface/HcalIsolatedTrackCandidateFwd.h"

#include <vector>
#include <map>
#include <utility>

namespace reco {
  
  class HcalIsolatedTrackCandidate: public RecoCandidate {

  public:
    
    // default constructor
    HcalIsolatedTrackCandidate() : RecoCandidate() { 
      maxP_   =-1;
      enEcal_ =-1;
      ptL1_   =etaL1_  =phiL1_=0;
      etaEcal_=phiEcal_=0;
      etaHcal_=phiHcal_=ietaHcal_=iphiHcal_=0;
      etaPhiEcal_=etaPhiHcal_=false; 
    }
    ///constructor from LorentzVector
    HcalIsolatedTrackCandidate(const LorentzVector& v): RecoCandidate(0,v) {
      maxP_   =-1;
      enEcal_ =-1;
      ptL1_   =etaL1_  =phiL1_=0;
      etaEcal_=phiEcal_=0;
      etaHcal_=phiHcal_=ietaHcal_=iphiHcal_=0;
      etaPhiEcal_=etaPhiHcal_=false; 
    }
    /// constructor from a track
    HcalIsolatedTrackCandidate(const reco::TrackRef & tr, double max, double ene): 
    RecoCandidate( 0, LorentzVector((tr.get()->px()),(tr.get())->py(),(tr.get())->pz(),(tr.get())->p()) ),
      track_(tr), maxP_(max), enEcal_(ene) {
      ptL1_   =etaL1_  =phiL1_=0;
      etaEcal_=phiEcal_=0;
      etaHcal_=phiHcal_=ietaHcal_=iphiHcal_=0;
      etaPhiEcal_=etaPhiHcal_=false; 
    }
    /// Copy constructor
    HcalIsolatedTrackCandidate(const HcalIsolatedTrackCandidate&);

    /// destructor
    virtual ~HcalIsolatedTrackCandidate();

    /// returns a clone of the candidate
    virtual HcalIsolatedTrackCandidate * clone() const;

    /// refrence to a Track
    virtual reco::TrackRef track() const;
    void setTrack(const reco::TrackRef & tr) { track_ = tr; }

    /// highest energy of other tracks in the cone around the candidate
    double maxP() const {return maxP_;}
    void SetMaxP(double mp) {maxP_=mp;}
          
    /// get reference to L1 jet
    virtual l1extra::L1JetParticleRef l1jet() const;
    void setL1Jet(const l1extra::L1JetParticleRef & jetRef) { l1Jet_ = jetRef; }
    std::pair<double,double> EtaPhiL1() const  {
      return std::pair<double,double>(etaL1_,phiL1_);
    }
    math::XYZTLorentzVector l1jetp() const;
    void setL1(double pt, double eta, double phi) {
      ptL1_ = pt; etaL1_ = eta; phiL1_ = phi;
    }

    /// ECAL energy in the inner cone around tau jet
    double energyEcal() const {return enEcal_;}
    void SetEnergyEcal(double a) {enEcal_=a;}

    ///eta, phi at ECAL surface
    void SetEtaPhiEcal(double eta, double phi) {
      etaEcal_=eta; phiEcal_=phi; etaPhiEcal_=true;
    }
    std::pair<double,double> EtaPhiEcal() const {
      return ((etaPhiEcal_) ? std::pair<double,double>(etaEcal_,phiEcal_) : std::pair<double,double>(0,0));
    }
    bool etaPhiEcal() const {return etaPhiEcal_;}

    ///eta, phi at HCAL surface
    void SetEtaPhiHcal(double eta, double phi, int ieta, int iphi) {
      etaHcal_=eta; phiHcal_=phi; ietaHcal_=ieta; iphiHcal_=iphi; etaPhiHcal_=true;
    }
    std::pair<double,double> EtaPhiHcal() const {
      return ((etaPhiHcal_) ? std::pair<double,double>(etaHcal_,phiHcal_) : std::pair<double,double>(0,0));
    }
    std::pair<int,int> towerIndex() const {
      return ((etaPhiHcal_) ? std::pair<int,int>(ietaHcal_,iphiHcal_) : std::pair<int,int>(0,0));
    }
    bool etaPhiHcal() const {return etaPhiHcal_;}

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a Track
    reco::TrackRef track_;
    /// reference to a L1 tau jet
    l1extra::L1JetParticleRef l1Jet_;
    /// highest P of other tracks in the cone around the candidate
    double maxP_;
    /// energy in ECAL around a cone around the track
    double enEcal_;
    /// pt, eta, phi of L1 object
    double ptL1_, etaL1_, phiL1_;
    /// eta, phi at ECAL
    bool   etaPhiEcal_;
    double etaEcal_, phiEcal_;
    /// eta, phi at HCAL
    bool   etaPhiHcal_;
    double etaHcal_, phiHcal_;
    int    ietaHcal_, iphiHcal_;
  };
  
  
}

#endif
