#ifndef TrackReco_TrackTrajectorySateOnDetInfos_h
#define TrackReco_TrackTrajectorySateOnDetInfos_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <vector>

namespace reco {

typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<reco::TrajectorySateOnDetInfoCollection> >  TrackTrajectorySateOnDetInfosCollection;
typedef  TrackTrajectorySateOnDetInfosCollection::value_type 						   TrackTrajectorySateOnDetInfos;
typedef  edm::Ref<TrackTrajectorySateOnDetInfosCollection> 						   TrackTrajectorySateOnDetInfosRef;
typedef  edm::RefProd<TrackTrajectorySateOnDetInfosCollection> 						   TrackTrajectorySateOnDetInfosRefProd;
typedef  edm::RefVector<TrackTrajectorySateOnDetInfosCollection> 					   TrackTrajectorySateOnDetInfosRefVector;

}

#endif
