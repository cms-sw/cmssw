#include "RecoMET/METProducers/interface/MinMETProducerT.h" 

#include "DataFormats/PatCandidates/interface/MET.h"

typedef MinMETProducerT<pat::MET> MinPatMETProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MinPatMETProducer);



