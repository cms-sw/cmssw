#include "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template class reco::RangeMapOwnVectorToVectorConverter<DTChamberId, DTRecSegment4D>;

using DTRecSegment4DOwnVectorRangeMapConverter = reco::RangeMapOwnVectorToVectorConverter<DTChamberId, DTRecSegment4D>;
DEFINE_FWK_MODULE(DTRecSegment4DOwnVectorRangeMapConverter);
