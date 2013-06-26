#include "FastSimDataFormats/External/interface/FastTrackerCluster.h"

FastTrackerCluster::FastTrackerCluster( const LocalPoint& pos      , 
					const LocalError& err      ,
					const DetId& id            ,
					const int simhitId         ,
					const int simtrackId       ,
					const uint32_t eeId        , 
					const float charge) :
  pos_(pos), err_(err), id_(id),
  simhitId_(simhitId) ,
  simtrackId_(simtrackId) ,
  eeId_(eeId), charge_(charge) {}




