#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::PhotonCollection> PhotonShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( PhotonShallowCloneProducer );
