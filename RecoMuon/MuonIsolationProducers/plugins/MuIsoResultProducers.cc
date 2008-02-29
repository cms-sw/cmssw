#include "MuIsolatorResultProducer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"

typedef MuIsolatorResultProducer<reco::Track> MuIsoTrackResultProducer;
typedef MuIsolatorResultProducer<reco::Candidate> MuIsoCandidateResultProducer;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuIsoTrackResultProducer);
DEFINE_FWK_MODULE(MuIsoCandidateResultProducer);
