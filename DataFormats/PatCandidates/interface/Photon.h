//
// $Id: Photon.h,v 1.4 2008/01/22 21:58:14 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Photon_h
#define DataFormats_PatCandidates_Photon_h

/**
  \class    Photon Photon.h "DataFormats/PatCandidates/interface/Photon.h"
  \brief    Analysis-level lepton class

   Photon implements the analysis-level charged lepton class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Photon.h,v 1.4 2008/01/22 21:58:14 lowette Exp $
*/

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace pat {

  typedef reco::Photon PhotonType;

  class Photon : public PATObject<PhotonType> {

    public:

      Photon();
      Photon(const PhotonType & aPhoton);
      Photon(const edm::Ref<std::vector<PhotonType> > & aPhotonRef);
      virtual ~Photon();

      const reco::Particle * genPhoton() const;

      void setGenPhoton(const reco::Particle & gl);

      float trackIso() const { return trackIso_; }       
      float caloIso()  const { return caloIso_; }       
      float photonID() const { return photonID_; }       

      void setTrackIso(float trackIso) { trackIso_ = trackIso; }       
      void setCaloIso(float caloIso)   { caloIso_ = caloIso; }       
      void setPhotonID(float photonID) { photonID_ = photonID; }       

    protected:

      std::vector<reco::Particle> genPhoton_;

      float trackIso_;
      float caloIso_;
      float photonID_;
  };



}

#endif
