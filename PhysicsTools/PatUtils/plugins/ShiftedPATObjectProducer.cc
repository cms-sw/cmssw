#include "PhysicsTools/PatUtils/interface/ShiftedParticleProducerT.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

typedef ShiftedParticleProducerT<pat::Electron> ShiftedPATElectronProducer;
typedef ShiftedParticleProducerT<pat::Photon> ShiftedPATPhotonProducer;
typedef ShiftedParticleProducerT<pat::Muon> ShiftedPATMuonProducer;
typedef ShiftedParticleProducerT<pat::Tau> ShiftedPATTauProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPATElectronProducer);
DEFINE_FWK_MODULE(ShiftedPATPhotonProducer);
DEFINE_FWK_MODULE(ShiftedPATMuonProducer);
DEFINE_FWK_MODULE(ShiftedPATTauProducer);
