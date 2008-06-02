#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::CaloMETCollection> CaloMETShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( CaloMETShallowCloneProducer );
