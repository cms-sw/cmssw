#include "JetMETCorrections/Type1MET/interface/PFJetMETcorrInputProducerT.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

typedef PFJetMETcorrInputProducerT<reco::PFJet, JetCorrExtractorT<reco::PFJet> > PFJetMETcorrInputProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFJetMETcorrInputProducer);

