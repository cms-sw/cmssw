#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <iostream>

#include "RecoParticleFlow/PFClusterTools/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationResultWrapper.h"
namespace pftools {
class Calibration {
public:
	
	Calibration();
	virtual ~Calibration();

	/*
	 * Pass in an input CalibrationResultWrapper, get the answer out.
	 * Not supported by all calibrators (not my best object design, I must confess).
	 */
	void calibrate(const CalibrationResultWrapper& crwInput,
			CalibrationResultWrapper& crwOutput) {
		calibrateCore(crwInput, crwOutput);
	}

	/* 
	 * More general calibrators may implement this function, which picks arbitrary fields
	 * from Calibratable in making its calibration.
	 */
	void calibrate(const Calibratable& c, CalibrationResultWrapper& crwOutput) {
		calibrateCore(c, crwOutput);
	}

	CalibrationProvenance getProvenance() const {
		return prov_;
	}

	CalibrationTarget getTarget() const {
		return targ_;
	}

protected:

	
	virtual void calibrateCore(const CalibrationResultWrapper& crwInput,
			CalibrationResultWrapper& crwOutput) {

		std::cout << __PRETTY_FUNCTION__
				<< ": WARNING: don't instantiate an instance of this class, but a subclass instead!"
				<< std::endl;
	}

	virtual void calibrateCore(const Calibratable& c,
			CalibrationResultWrapper& crwOutput) {
		std::cout << __PRETTY_FUNCTION__
				<< ": WARNING: don't instantiate an instance of this class, but a subclass instead!"
				<< std::endl;
	}

	CalibrationProvenance prov_;
	CalibrationTarget targ_;
	
private:
	
};
}

#endif /*CALIBRATION_H_*/
