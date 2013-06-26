#ifndef CALIBRATIONPROVENANCE_H_
#define CALIBRATIONPROVENANCE_H_

namespace pftools {
/*
 * \class CalibrationProvenance
 * \brief Enumerates possible calibrators and their results
 * 
 * \author Jamie Ballin
 * \date June 2008
 */
enum CalibrationProvenance {
	UNCALIBRATED = 0, LINEAR = 1, BAYESIAN = 2, LINEARECAL = 3, LINEARHCAL=4, LINEARCORR = -1, NONE = 99
};

enum CalibrationTarget {
        UNDEFINED = 0, CLUSTER = 1, RECHIT = 2, PFCANDIDATE = 3, PFELEMENT = 4, CORR_CLUSTER = -1,
PRE_RECHIT = 6, PRE_PFCANDIDATE = 7
};


}
#endif /*CALIBRATIONPROVENANCE_H_*/
