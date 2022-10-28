#include "CommonTools/UtilAlgos/interface/FwdPtrProducer.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

typedef edm::FwdPtrProducer<reco::PFJet> PFJetFwdPtrProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetFwdPtrProducer);
