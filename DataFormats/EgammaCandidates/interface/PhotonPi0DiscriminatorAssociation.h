#ifndef EgammaReco_PhotonPi0DiscriminatorAssociation_h
#define EgammaReco_PhotonPi0DiscriminatorAssociation_h
//  \class PhotonPi0DiscriminatorAssociation
//
//  \association of Pi0Discriminator to PhotonCollection
//
//

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::Photon>, float> > PhotonPi0DiscriminatorAssociationMap;
}

#endif
