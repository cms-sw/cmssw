#ifndef __CINT__
#ifndef MatrixFillMap_H
#define MatrixFillMap_H

#include "Calibration/EcalCalibAlgos/interface/VFillMap.h"

class MatrixFillMap : public VFillMap
{
	public:
	//! ctor
	MatrixFillMap (int ,
		       int ,
			const std::map<int,int>& ,
			double,
			double, 
			const std::map<int,int>&,
			EcalIntercalibConstantMap *,
			EcalIntercalibConstantMap *);
	//! dtor
	~MatrixFillMap ();
        void fillMap (const std::vector<std::pair<DetId,float> > &, 
			const DetId,
			const EcalRecHitCollection *, 
			const EcalRecHitCollection *,
			std::map<int,double> & xtlMap,
			double & ) ;
	private:
	//! takes care of the Barrel
        void fillEBMap (EBDetId , 
			const EcalRecHitCollection *, 
			std::map<int, double> & , 
			int , double & ) ;

	//! takes care of the Endcap
        void fillEEMap (EEDetId ,
			const EcalRecHitCollection *, 
			std::map<int, double> & , 
			int , double & ) ;

};
#endif
#endif
