#include "JetMETCorrections/Type1MET/interface/CaloJetMETcorrInputProducerT.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

typedef CaloJetMETcorrInputProducerT<reco::CaloJet> CaloJetMETcorrInputProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CaloJetMETcorrInputProducer);
