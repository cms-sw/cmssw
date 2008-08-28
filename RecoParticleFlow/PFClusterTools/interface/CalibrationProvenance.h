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

}
#endif /*CALIBRATIONPROVENANCE_H_*/
