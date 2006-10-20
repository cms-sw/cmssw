#ifndef EgammaCandidates_PhotonIsolationAssociation_h
#define EgammaCandidates_PhotonIsolationAssociation_h
// \class PhotonIsolationAssociation
// 
// \short association of Isolation to a Photon
// $Id: $
//

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::Photon>, float > > PhotonIsolationMap;
}
#endif
