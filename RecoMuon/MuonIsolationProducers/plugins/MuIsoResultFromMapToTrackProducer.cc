#include "MuIsolatorResultProducer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"

typedef MuIsolatorResultProducer<reco::MuIsoDepositAssociationMap> MuIsoResultFromMapToTrackProducer;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuIsoResultFromMapToTrackProducer);
