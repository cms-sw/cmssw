#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/Modules/interface/CorrectedJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"


using namespace reco;

typedef CorrectedJetProducer<CaloJet> CorrectedCaloJetProducer;
DEFINE_FWK_MODULE(CorrectedCaloJetProducer);

typedef CorrectedJetProducer<PFJet> CorrectedPFJetProducer;
DEFINE_FWK_MODULE(CorrectedPFJetProducer);

typedef CorrectedJetProducer<JPTJet> CorrectedJPTJetProducer;
DEFINE_FWK_MODULE(CorrectedJPTJetProducer);

typedef CorrectedJetProducer<TrackJet> CorrectedTrackJetProducer;
DEFINE_FWK_MODULE(CorrectedTrackJetProducer);

typedef CorrectedJetProducer<GenJet> CorrectedGenJetProducer;
DEFINE_FWK_MODULE(CorrectedGenJetProducer);
