#ifndef LINEARCALIBRATION_H_
#define LINEARCALIBRATION_H_

#include "RecoParticleFlow/PFClusterTools/interface/Calibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationResultWrapper.h"
#include <iostream>

namespace pftools {
class LinearCalibration : public pftools::Calibration {
public:
	/* If a == 0, no offset element is included
	 * If b_ == 0, it's HCAL only
	 * If c_ == 0, it's ECAL only
	 */
	double a_, b_, c_;

	LinearCalibration();
	LinearCalibration(CalibrationTarget t, double b, double c);
	LinearCalibration(CalibrationTarget t, double a, double b, double c);



	virtual ~LinearCalibration();
protected:
	virtual void calibrateCore(const CalibrationResultWrapper& crwInput,
			CalibrationResultWrapper& crwOutput);


};

std::ostream& operator<<(std::ostream& s, const LinearCalibration& lc);

}

#endif /*LINEARCALIBRATION_H_*/
