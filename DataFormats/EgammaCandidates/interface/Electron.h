#ifndef EgammaReco_Electron_h
#define EgammaReco_Electron_h
/** \class reco::Electron
 *
 * Reconstructed Electron with references 
 * to a SuperCluster
 * and a Track
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Electron.h,v 1.7 2006/03/01 13:51:27 llista Exp $
 *
 */
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

namespace reco {

  class SuperCluster;
  class Track;

  class Electron {
  public:
    /// spatial vector
    typedef math::RhoEtaPhiVector Vector;
    /// default constructor
    Electron() { }
    /// constructor from values
    Electron( const Vector & calo, const Vector & track, short charge,
	      double eHadOverEcal, short isolation, short pixelLines );
    /// constructor from references to a SuperCluster and a Track
    Electron( const SuperClusterRef & calo, TrackRef track,
	      double eHadOverEcal, short isolation, short pixelLines );
    /// momentum vector from calorimeter
    const Vector & caloMomentum() const { return caloMomentum_; }
    /// transverse energy from calorimeter
    double caloEt() const { return caloMomentum_.Rho(); }
    /// preudorapidity from calorimeter
    double caloEta() const { return caloMomentum_.Eta(); }
    /// azimuthal angle from calorimeter
    double caloPhi() const { return caloMomentum_.Phi(); }
    /// polar angle angle from calorimeter
    double caloTheta() const { return caloMomentum_.Theta(); }
    /// energy measured in the calorimeter
    double caloEnergy() const { return caloMomentum_.R(); }
    /// momentum vector from Track
    const Vector & trackMomentum() const { return trackMomentum_; }
    /// transverse momentum from Track
    double trackPt() const { return trackMomentum_.Rho(); }
    /// pseudorapidity momentum from Track
    double trackEta() const { return trackMomentum_.Eta(); }
    /// azimuthal angle momentum from Track
    double trackPhi() const { return trackMomentum_.Phi(); }
    /// polar angle momentum from Track
    double trackTheta() const { return trackMomentum_.Theta(); }
    /// magnitude of momentum vector from Track
    double trackP() const { return trackMomentum_.R(); }
    /// ratio of energy measured in the calorimeter and Track momentum
    double eOverP() const { return caloEnergy() / trackP(); }
    /// electron electric charge
    short charge() const { return charge_; }
    /// ratio of energy deposits in Hcal over Ecal
    double eHadOverEcal() const { return eHadOverEcal_; }
    /// difference of pseudorapidity in calorimeter and Track
    double deltaEta() const { return  caloEta() - trackEta(); }
    /// isolation (should be better documented!)
    short isolation() const { return isolation_; }
    /// number of pixel lines (should be better documented!)
    short pixelLines() const { return pixelLines_; }
    /// reference to SuperCluster
    const SuperClusterRef & superCluster() const { return superCluster_; }
    /// set reference to SuperCluster
    void setSuperCluster( const SuperClusterRef & c ) { superCluster_ = c; }
    /// reference to Track
    const TrackRef & track() const { return track_; }
    /// set reference to Track
    void setTrack( const TrackRef & t ) { track_ = t; }

  private:
    /// momentum vector from calorimeter
    Vector caloMomentum_;
    /// momentum vector from Track
    Vector trackMomentum_;
    /// electric charge
    short charge_;
    /// ratio of energy deposits in Hcal over Ecal
    Double32_t eHadOverEcal_;
    /// isolation (should be better documented!)
    short isolation_;
     /// number of pixel lines (should be better documented!)
    short pixelLines_;
    /// reference to SuperCluster
    SuperClusterRef superCluster_;
    /// reference to Track
    TrackRef track_;
  };
}

#endif
