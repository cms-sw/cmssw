#include "Calibration/EcalCalibAlgos/interface/VFillMap.h"

	VFillMap::VFillMap (int WindowX,int WindowY,
			const std::map<int,int>& xtalReg,double minE,
			double maxE, const std::map<int,int>& RingReg,
			EcalIntercalibConstantMap * barrelMap,
			EcalIntercalibConstantMap * endcapMap):
	m_recoWindowSidex (WindowX),
	m_recoWindowSidey (WindowY),
        m_xtalRegionId (xtalReg),
        m_minEnergyPerCrystal (minE),
        m_maxEnergyPerCrystal (maxE),
        m_IndexInRegion (RingReg),
        m_barrelMap (barrelMap),
        m_endcapMap (endcapMap)
	
	{}
