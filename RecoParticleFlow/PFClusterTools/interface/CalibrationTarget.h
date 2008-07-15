#ifndef CALIBRATIONTARGET_H_
#define CALIBRATIONTARGET_H_

/*
 * \class CalibrationTarget
 * \brief Enumerates possible targets for calibration
 * 
 * The undefined element means just that!
 * 
 * \author Jamie Ballin
 * \date June 2008
 */
enum CalibrationTarget {
	UNDEFINED = 0, CLUSTER = 1, RECHIT = 2, PFCANDIDATE = 3, PFELEMENT = 4, PRE_CLUSTER = 5, PRE_RECHIT = 6, PRE_PFCANDIDATE = 7
};

#endif /*CALIBRATIONTARGET_H_*/
