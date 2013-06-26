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
        void fillMap (const std::vector<std::pair<DetId,float> > &, 
			const DetId,
			const EcalRecHitCollection *, 
			const EcalRecHitCollection *,
			std::map<int,double> & xtlMap,
			double & ) ;
};
#endif
#endif
