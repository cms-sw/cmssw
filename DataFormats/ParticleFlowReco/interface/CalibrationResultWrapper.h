#ifndef CALIBRATIONRESULTWRAPPER_H_
#define CALIBRATIONRESULTWRAPPER_H_

//#include <boost/shared_ptr.hpp>

#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"

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

//	typedef boost::shared_ptr<CalibrationResultWrapper>
//			CalibrationResultWrapperPtr;

	CalibrationResultWrapper() {
		reset();
	}

	virtual ~CalibrationResultWrapper() {
	}


	void reset() {
		resetCore();
	}


	void compute() {
		computeCore();
	}

	double bias() const {
		return (particleEnergy_ -  truthEnergy_) / truthEnergy_;
	}

	double ratio() const {
		return(particleEnergy_/truthEnergy_);
	}

	/*
	 * Which calibrator made this?
	 */
	CalibrationProvenance provenance_;
	/*
	 * What energy was this particle optimised to?
	 */
	double truthEnergy_;

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

	/*
	 * (reco - truth)/truth
	 */
	double bias_;

	/*
	 * reco/truth
	 */
	double ratio_;

	/*
	* Target function contribution
	*/
	double targetFuncContrib_;

	double a_;
	double b_;
	double c_;

private:

	virtual void computeCore() {
		bias_ = bias();
		ratio_ = ratio();
	}

	virtual void resetCore() {
		truthEnergy_ = 0;
		ecalEnergy_ = 0;
		hcalEnergy_ = 0;
		particleEnergy_ = 0;
		provenance_ = UNCALIBRATED;
		target_ = UNDEFINED;
		bias_ = 0;
		ratio_ = 1.0;
		targetFuncContrib_ = 0;
		a_ = 0.0;
		b_ = 1.0;
		c_ = 1.0;
	}

};

}

#endif /*CALIBRATIONRESULTWRAPPER_H_*/
