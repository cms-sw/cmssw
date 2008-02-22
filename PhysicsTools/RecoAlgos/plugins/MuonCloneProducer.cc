#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "PhysicsTools/CandAlgos/interface/CloneProducer.h"

typedef CloneProducer<reco::MuonCollection> MuonCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MuonCloneProducer );
