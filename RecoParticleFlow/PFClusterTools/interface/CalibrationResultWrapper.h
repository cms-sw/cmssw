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
	

	void reset() {
		resetCore();
	}
	
	
	void compute() {
		computeCore();
	}
	
	double bias() const {
		return (particleEnergy_ -  truthEnergy_) / truthEnergy_;
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
	
	double bias_;

private:
	
	virtual void computeCore() {
		bias_ = bias();
	}
	
	virtual void resetCore() {
		truthEnergy_ = 0;
		ecalEnergy_ = 0;
		hcalEnergy_ = 0;
		particleEnergy_ = 0;
		provenance_ = UNCALIBRATED;
		target_ = UNDEFINED;
		bias_ = 0;
	}

};

}

#endif /*CALIBRATIONRESULTWRAPPER_H_*/
