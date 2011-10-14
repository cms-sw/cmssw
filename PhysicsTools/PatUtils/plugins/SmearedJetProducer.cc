#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

typedef SmearedJetProducerT<reco::CaloJet> SmearedCaloJetProducer;
typedef SmearedJetProducerT<reco::PFJet> SmearedPFJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SmearedCaloJetProducer);
DEFINE_FWK_MODULE(SmearedPFJetProducer);
