#include "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template class reco::RangeMapOwnVectorToVectorConverter<CSCDetId, CSCSegment>;

using CSCSegmentOwnVectorRangeMapConverter = reco::RangeMapOwnVectorToVectorConverter<CSCDetId, CSCSegment>;
DEFINE_FWK_MODULE(CSCSegmentOwnVectorRangeMapConverter);
