//
// $Id: Photon.h,v 1.4 2008/02/07 18:16:13 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Photon_h
#define DataFormats_PatCandidates_Photon_h

/**
  \class    pat::Photon Photon.h "DataFormats/PatCandidates/interface/Photon.h"
  \brief    Analysis-level Photon class

   Photon implements the analysis-level photon class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Photon.h,v 1.4 2008/02/07 18:16:13 lowette Exp $
*/

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


namespace pat {


  typedef reco::Photon PhotonType;


  class Photon : public PATObject<PhotonType> {

    public:

      Photon();
      Photon(const PhotonType & aPhoton);
      Photon(const edm::RefToBase<PhotonType> & aPhotonRef);
      virtual ~Photon();

      const reco::Particle * genPhoton() const;

      void setGenPhoton(const reco::Particle & gp);

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
