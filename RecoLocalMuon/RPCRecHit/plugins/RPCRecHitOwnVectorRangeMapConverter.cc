#include "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template class reco::RangeMapOwnVectorToVectorConverter<RPCDetId, RPCRecHit>;

using RPCRecHitOwnVectorRangeMapConverter = reco::RangeMapOwnVectorToVectorConverter<RPCDetId, RPCRecHit>;
DEFINE_FWK_MODULE(RPCRecHitOwnVectorRangeMapConverter);
