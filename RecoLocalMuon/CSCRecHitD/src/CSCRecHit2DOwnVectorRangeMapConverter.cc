#include "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template class reco::RangeMapOwnVectorToVectorConverter<CSCDetId, CSCRecHit2D>;

using CSCRecHit2DOwnVectorRangeMapConverter = reco::RangeMapOwnVectorToVectorConverter<CSCDetId, CSCRecHit2D>;
DEFINE_FWK_MODULE(CSCRecHit2DOwnVectorRangeMapConverter);
