#ifndef CALIBRATIONRESULTWRAPPER_H_
#define CALIBRATIONRESULTWRAPPER_H_

#include <boost/shared_ptr.hpp>

#include "RecoParticleFlow/PFClusterTools/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"

namespace pftools {

/**
 * \class CalibrationResultWrapper
 * \brief A small class designed to hold the result of a calibration of a SingleParticleWrapper
 * 
 * \author Jamie Ballin
 * \date May 2008
 *
 */ 
class CalibrationResultWrapper {
public:

	typedef boost::shared_ptr<CalibrationResultWrapper>
			CalibrationResultWrapperPtr;

	CalibrationResultWrapper() {
		reset();
	}

	virtual ~CalibrationResultWrapper() {
	}
	;

	void reset() {
		resetCore();
	}
	;
	
	double bias() {
		return (particleEnergy_ -  truthEnergy_) / truthEnergy_;
	}
	
	/*
	 * Which calibrator made this?
	 */
	CalibrationProvenance provenance_;
	/*
	 * What energy was this particle optimised to?
	 */
	double truthEnergy_ = 0;
	
	/*
	 * Calibrated ecal deposition
	 */
	double ecalEnergy_;
	
	/*
	 * Calibrated hcal deposition
	 */
	double hcalEnergy_;
	
	/*
	 * Calibrated particle energy (not necessarily ecal + hcal!)
	 */
	double particleEnergy_;
	
	/*
	 * What objects did this optimise on?
	 */
	CalibrationTarget target_;

private:
	virtual void resetCore() {
		truthEnergy_ = 0;
		ecalEnergy_ = 0;
		hcalEnergy_ = 0;
		particleEnergy_ = 0;
		provenance_ = UNCALIBRATED;
		target_ = UNDEFINED;
	}

};

}

#endif /*CALIBRATIONRESULTWRAPPER_H_*/
