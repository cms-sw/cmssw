#ifndef CALIBRATIONRECORD_H_
#define CALIBRATIONRECORD_H_

namespace pftools {
/*
 * \class CalibrationRecord
 * \brief An object which maintains a record of what parameters a calibrator needs to make a calibration
 * 
 * \author Jamie Ballin
 * \date May 2008
 */
class CalibrationRecord {
public:
	CalibrationRecord();
	virtual ~CalibrationRecord();
};
}
#endif /*CALIBRATIONRECORD_H_*/
