#ifndef EgammaReco_Photon_h
#define EgammaReco_Photon_h
/** \class reco::Photon
 *  
 * Reconstructed Photon with reference
 * to a SuperCluster
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.7 2006/03/08 11:16:35 llista Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

namespace reco {
  class Photon {
  public:
    /// spatial vector
    typedef math::RhoEtaPhiVector Vector;
    /// default constructor
    Photon() { }
    /// constructor from values
    Photon( const Vector &,
	    double z, short isolation, short pixelLines, bool hasSeed );
    /// momentum vector
    Vector momentum() const { return momentum_; }
    /// energy
    double energy() const { return momentum_.R(); }
    /// tranverse momentum
    double pt() const { return momentum_.Rho(); }
    /// pseudorapidity
    double eta() const { return momentum_.Eta(); }
    /// azimuthal angle
    double phi() const { return momentum_.Phi(); }
    /// polar angle
    double theta() const { return momentum_.Theta(); }
    /// momentum magnitude
    double p() const { return momentum_.R(); }
    /// x coodrinate of momentum vector
    double px() const { return momentum_.X(); }
    /// y coodrinate of momentum vector
    double py() const { return momentum_.Y(); }
    /// z coodrinate of momentum vector
    double pz() const { return momentum_.Z(); }
    /// z coordinate of assumed vertex
    double vtxZ() const { return vtxZ_; }
    /// isolation (should be better documented!)
    short isolation() const { return isolation_; }
    /// pixel lines (should be better documented!)
    short pixelLines() const { return pixelLines_; }
    /// returns true if has a seed
    bool hasSeed() const { return hasSeed_; }
    /// reference to SuperCluster
    const SuperClusterRef & superCluster() const { return superCluster_; }
    /// set reference to SuperCluster
    void setSuperCluster( const SuperClusterRef & c ) { superCluster_ = c; }

  private:
    /// momentum vector
    Vector momentum_;
    ///  z coordinate of assumed vertex
    Double32_t vtxZ_;
    /// isolation (should be better documented!)
    short isolation_;
    /// pixel lines (should be better documented!)
    short pixelLines_;
    /// true if has a seed
    bool hasSeed_;
    /// reference to SuperCluster
    SuperClusterRef superCluster_;
  };
}

#endif
