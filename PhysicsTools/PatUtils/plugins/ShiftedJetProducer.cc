#include "PhysicsTools/PatUtils/interface/ShiftedJetProducerT.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

typedef ShiftedJetProducerT<reco::CaloJet, JetCorrExtractorT<reco::CaloJet> > ShiftedCaloJetProducer;
typedef ShiftedJetProducerT<reco::PFJet, JetCorrExtractorT<reco::PFJet> > ShiftedPFJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedCaloJetProducer);
DEFINE_FWK_MODULE(ShiftedPFJetProducer);
