#ifndef TrackReco_TrackDeDxEstimate_h
#define TrackReco_TrackDeDxEstimate_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <vector>


namespace reco {
//Association Track -> float estimator
typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<float> >  TrackDeDxEstimateCollection;
typedef  TrackDeDxEstimateCollection::value_type TrackDeDxEstimate;
typedef  edm::Ref<TrackDeDxEstimateCollection> TrackDeDxEstimateRef;
typedef  edm::RefProd<TrackDeDxEstimateCollection> TrackDeDxEstimateRefProd;
typedef  edm::RefVector<TrackDeDxEstimateCollection> TrackDeDxEstimateRefVector;
}
#endif
