#ifndef __PFBlockElementSuperCluster__
#define __PFBlockElementSuperCluster__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

namespace reco {
  
  /// \brief Cluster Element.
  /// 
  /// this class contains a reference to a PFCluster 
  class PFBlockElementSuperCluster : public PFBlockElement {
  public:
    PFBlockElementSuperCluster() {} 
    
    /// \brief constructor.
    /// type must be equal to PS1, PS2, ECAL, HCAL. 
    /// \todo add a protection against the other types...
    PFBlockElementSuperCluster(const SuperClusterRef& ref) 
      : 
      PFBlockElement(PFBlockElement::SC),
      superClusterRef_( ref ),
      trackIso_(0.),
      ecalIso_(0.),
      hcalIso_(0.),
      HoE_(0.),
      fromGsfElectron_(false),
      fromPhoton_(false),
      fromPFSuperCluster_(false){}
      
    PFBlockElement* clone() const { return new PFBlockElementSuperCluster(*this); }
    
    /// \return reference to the corresponding cluster
    SuperClusterRef  superClusterRef() const {return superClusterRef_;}

    /// \return reference to seeding photon
    PhotonRef photonRef() const {return photonRef_;}

    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;

    /// set the track Iso
    void setTrackIso(float val) {trackIso_=val;}

    /// set the ecal Iso
    void setEcalIso(float val) {ecalIso_=val;}

    /// set the had Iso
    void setHcalIso(float val) {hcalIso_=val;}

    /// set H/E
    void setHoE(float val) {HoE_=val;}

    /// set provenance
    void setFromGsfElectron(bool val) {fromGsfElectron_=val;}

    /// set provenance
    void setFromPhoton(bool val) {fromPhoton_=val;}

    void setFromPFSuperCluster(bool val) { fromPFSuperCluster_ = val; }
    
    /// set photonRef
    void setPhotonRef(const PhotonRef & ref) {photonRef_ = ref ;}

    /// \return the track isolation
    float trackIso() const {return trackIso_;}

    /// \return the ecal isolation
    float ecalIso() const {return ecalIso_;}
    
    /// \return the had isolation
    float hcalIso() const {return hcalIso_;}

    /// \return Hoe
    float hoverE() const {return HoE_;}

    /// \return provenance
    bool fromGsfElectron() const {return fromGsfElectron_;}

    /// \return provenance
    bool fromPhoton() const {return fromPhoton_;}
    
    //SuperCluster comes from a PFSuperCluster (and can therefore be matched
    // by ref back to the initial PFClusters)
    bool fromPFSuperCluster() const { return fromPFSuperCluster_; }

  private:
    /// reference to the corresponding cluster
    SuperClusterRef  superClusterRef_;
    PhotonRef photonRef_;

    float trackIso_;
    float ecalIso_;
    float hcalIso_;
    float HoE_;

    bool fromGsfElectron_;
    bool fromPhoton_;
    bool fromPFSuperCluster_;
  };
}

#endif

