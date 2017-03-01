#include "JetMETCorrections/Type1MET/interface/JetCleanerForType1METT.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

typedef JetCleanerForType1METT<reco::PFJet, JetCorrExtractorT<reco::PFJet> > PFJetCleanerForType1MET;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFJetCleanerForType1MET);

