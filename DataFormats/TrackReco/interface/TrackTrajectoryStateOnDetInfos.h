#ifndef TrackReco_TrackTrajectoryStateOnDetInfos_h
#define TrackReco_TrackTrajectoryStateOnDetInfos_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrajectoryStateOnDetInfo.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <vector>

namespace reco {

//typedef  edm::AssociationMap<  edm::OneToValue<reco::TrackCollection, reco::TrajectoryStateOnDetInfoCollection >     > TrackTrajectoryStateOnDetInfosCollection;
typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<reco::TrajectoryStateOnDetInfoCollection> >  TrackTrajectoryStateOnDetInfosCollection;
typedef  TrackTrajectoryStateOnDetInfosCollection::value_type 					  	    TrackTrajectoryStateOnDetInfos;
typedef  edm::Ref<TrackTrajectoryStateOnDetInfosCollection> 						    TrackTrajectoryStateOnDetInfosRef;
typedef  edm::RefProd<TrackTrajectoryStateOnDetInfosCollection> 					    TrackTrajectoryStateOnDetInfosRefProd;
typedef  edm::RefVector<TrackTrajectoryStateOnDetInfosCollection> 					    TrackTrajectoryStateOnDetInfosRefVector;

}

#endif
