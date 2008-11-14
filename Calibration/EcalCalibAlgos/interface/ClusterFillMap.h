#ifndef __CINT__
#ifndef ClusterFillMap_H
#define ClusterFillMap_H

#include "Calibration/EcalCalibAlgos/interface/VFillMap.h"

class ClusterFillMap : public VFillMap
{
	public:
	//!ctor
	ClusterFillMap (int ,
			int ,
			std::map<int,int> ,
			double,
			double, 
		        std::map<int,int>,
  			EcalIntercalibConstantMap *,
			EcalIntercalibConstantMap *);
	
	//!dtor
	~ClusterFillMap ();

	//!Fills the map
        DetId fillMap (const std::vector<DetId> &, 
			const EcalRecHitCollection *, 
			const EcalRecHitCollection *,
			std::map<int,double> & xtlMap,
			double & ) ;
};
#endif
#endif
