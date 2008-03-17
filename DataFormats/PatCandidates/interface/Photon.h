//
// $Id: Photon.h,v 1.5 2008/03/05 14:47:33 fronga Exp $
//

#ifndef DataFormats_PatCandidates_Photon_h
#define DataFormats_PatCandidates_Photon_h

/**
  \class    pat::Photon Photon.h "DataFormats/PatCandidates/interface/Photon.h"
  \brief    Analysis-level Photon class

   Photon implements the analysis-level photon class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Photon.h,v 1.5 2008/03/05 14:47:33 fronga Exp $
*/

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"


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

      float photonID() const { return photonID_; }
      void setPhotonID(float photonID) { photonID_ = photonID; }

    protected:

      std::vector<reco::Particle> genPhoton_;

      float photonID_;
#include "DataFormats/PatCandidates/interface/isolation_impl.h"
  };


}

#endif
