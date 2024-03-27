#include "CommonTools/RecoAlgos/interface/RangeMapOwnVectorToVectorConverter.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

template class reco::RangeMapOwnVectorToVectorConverter<GEMDetId, GEMRecHit>;

using GEMRecHitOwnVectorRangeMapConverter = reco::RangeMapOwnVectorToVectorConverter<GEMDetId, GEMRecHit>;
DEFINE_FWK_MODULE(GEMRecHitOwnVectorRangeMapConverter);
