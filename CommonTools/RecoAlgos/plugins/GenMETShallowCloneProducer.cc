#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "CommonTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::GenMETCollection> GenMETShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenMETShallowCloneProducer );
