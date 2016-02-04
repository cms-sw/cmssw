#ifndef TrackReco_TrackDeDxHits_h
#define TrackReco_TrackDeDxHits_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <vector>

namespace reco {

// Association Track -> DeDx hits   
typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<reco::DeDxHitCollection> >  TrackDeDxHitsCollection;
typedef  TrackDeDxHitsCollection::value_type TrackDeDxHits;
typedef  edm::Ref<TrackDeDxHitsCollection> TrackDeDxHitsRef;
typedef  edm::RefProd<TrackDeDxHitsCollection> TrackDeDxHitsRefProd;
typedef  edm::RefVector<TrackDeDxHitsCollection> TrackDeDxHitsRefVector;

}

#endif
