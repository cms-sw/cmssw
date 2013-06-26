#ifndef MuonReco_CaloMuon_h
#define MuonReco_CaloMuon_h
/** \class reco::CaloMuon CaloMuon.h DataFormats/MuonReco/interface/CaloMuon.h
 *  
 * A lightweight reconstructed Muon to store low momentum muons without matches
 * in the muon detectors. Contains:
 *  - reference to a silicon tracker track
 *  - calorimeter energy deposition
 *  - calo compatibility variable
 *
 * \author Dmytro Kovalskyi, UCSB
 *
 * \version $Id: CaloMuon.h,v 1.4 2009/03/15 03:33:32 dmytro Exp $
 *
 */
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {
 
  class CaloMuon {
  public:
    CaloMuon();
    virtual ~CaloMuon(){}     
    
    /// reference to Track reconstructed in the tracker only
    virtual TrackRef innerTrack() const { return innerTrack_; }
    virtual TrackRef track() const { return innerTrack(); }
    /// set reference to Track
    virtual void setInnerTrack( const TrackRef & t ) { innerTrack_ = t; }
    virtual void setTrack( const TrackRef & t ) { setInnerTrack(t); }
    /// energy deposition
    bool isEnergyValid() const { return energyValid_; }
    /// get energy deposition information
    MuonEnergy calEnergy() const { return calEnergy_; }
    /// set energy deposition information
    void setCalEnergy( const MuonEnergy& calEnergy ) { calEnergy_ = calEnergy; energyValid_ = true; }
     
    /// Muon hypothesis compatibility block
    /// Relative likelihood based on ECAL, HCAL, HO energy defined as
    /// L_muon/(L_muon+L_not_muon)
    float caloCompatibility() const { return caloCompatibility_; }
    void  setCaloCompatibility(float input){ caloCompatibility_ = input; }
    bool  isCaloCompatibilityValid() const { return caloCompatibility_>=0; } 
     
    /// a bunch of useful accessors
    int charge() const { return innerTrack_.get()->charge(); }
    /// polar angle  
    double theta() const { return innerTrack_.get()->theta(); }
    /// momentum vector magnitude
    double p() const { return innerTrack_.get()->p(); }
    /// track transverse momentum
    double pt() const { return innerTrack_.get()->pt(); }
    /// x coordinate of momentum vector
    double px() const { return innerTrack_.get()->px(); }
    /// y coordinate of momentum vector
    double py() const { return innerTrack_.get()->py(); }
    /// z coordinate of momentum vector
    double pz() const { return innerTrack_.get()->pz(); }
    /// azimuthal angle of momentum vector
    double phi() const { return innerTrack_.get()->phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return innerTrack_.get()->eta(); }
     
  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef innerTrack_;
    /// energy deposition 
    MuonEnergy calEnergy_;
    bool energyValid_;
    /// muon hypothesis compatibility with observer calorimeter energy
    float caloCompatibility_;
  };

}


#endif
