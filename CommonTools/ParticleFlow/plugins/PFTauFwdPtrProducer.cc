#include "CommonTools/UtilAlgos/interface/FwdPtrProducer.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

typedef edm::FwdPtrProducer<reco::PFTau> PFTauFwdPtrProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTauFwdPtrProducer);
