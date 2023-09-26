#include "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template class reco::RangeMapOwnVectorToVectorConverter<GEMDetId, GEMSegment>;

using GEMSegmentOwnVectorRangeMapConverter = reco::RangeMapOwnVectorToVectorConverter<GEMDetId, GEMSegment>;
DEFINE_FWK_MODULE(GEMSegmentOwnVectorRangeMapConverter);
