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
 * \version $Id: CaloMuon.h,v 1.2 2008/04/30 18:17:44 dmytro Exp $
 *
 */
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {
 
  class CaloMuon {
  public:
    CaloMuon();
    
    /// reference to Track reconstructed in the tracker only
    TrackRef track() const { return track_; }
    /// set reference to Track
    void setTrack( const TrackRef & t ) { track_ = t; }
    /// energy deposition
    bool isEnergyValid() const { return energyValid_; }
    /// get energy deposition information
    MuonEnergy calEnergy() const { return calEnergy_; }
    MuonEnergy getCalEnergy() const __attribute__((deprecated));
    /// set energy deposition information
    void setCalEnergy( const MuonEnergy& calEnergy ) { calEnergy_ = calEnergy; energyValid_ = true; }
     
    /// Muon hypothesis compatibility block
    /// Relative likelihood based on ECAL, HCAL, HO energy defined as
    /// L_muon/(L_muon+L_not_muon)
    float caloCompatibility() const { return caloCompatibility_; }
    float getCaloCompatibility() const __attribute__((deprecated));
    void  setCaloCompatibility(float input){ caloCompatibility_ = input; }
    bool  isCaloCompatibilityValid() const { return caloCompatibility_>=0; } 
     
    /// a bunch of useful accessors
    int charge() const { return track_.get()->charge(); }
    /// polar angle  
    double theta() const { return track_.get()->theta(); }
    /// momentum vector magnitude
    double p() const { return track_.get()->p(); }
    /// track transverse momentum
    double pt() const { return track_.get()->pt(); }
    /// x coordinate of momentum vector
    double px() const { return track_.get()->px(); }
    /// y coordinate of momentum vector
    double py() const { return track_.get()->py(); }
    /// z coordinate of momentum vector
    double pz() const { return track_.get()->pz(); }
    /// azimuthal angle of momentum vector
    double phi() const { return track_.get()->phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return track_.get()->eta(); }
     
  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef track_;
    /// energy deposition 
    MuonEnergy calEnergy_;
    bool energyValid_;
    /// muon hypothesis compatibility with observer calorimeter energy
    float caloCompatibility_;
  };

}


#endif
