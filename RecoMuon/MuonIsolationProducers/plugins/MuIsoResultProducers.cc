#include "MuIsolatorResultProducer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

typedef MuIsolatorResultProducer<reco::Track> MuIsoTrackResultProducer;
typedef MuIsolatorResultProducer<reco::Candidate> MuIsoCandidateResultProducer;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuIsoTrackResultProducer);
DEFINE_FWK_MODULE(MuIsoCandidateResultProducer);
