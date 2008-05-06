#ifndef CALIBRATOR_HH_
#define CALIBRATOR_HH_

#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.hh"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.hh"

#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace pftools {

/**
 \class Calibrator 
 \brief Abstract base class for Particle Flow calibration algorithms

 \author Jamie Ballin
 \date   April 2008
 */
class Calibrator {
public:

	Calibrator();
	virtual ~Calibrator();

	void addDetectorElement(DetectorElementPtr const de);

	void addParticleDeposit(ParticleDepositPtr pd);

	/*
	 * Returns the calibration coefficient for each detector element, using data
	 * from all particle depositions stored within.
	 */
	std::map<DetectorElementPtr, double> getCalibrationCoefficients() throw(
			PFToolsException&) {
		return getCalibrationCoefficientsCore();
	}
	;

//	DetectorElementPtr getOffsetElement() {
//		return getOffsetElementCore();
//	}
//	;

	/* 
	 * Here we use the virtual constructor idea to allow for plug-and-play Calibrators
	 * See http://www.parashift.com/c++-faq-lite/virtual-functions.html#faq-20.8
	 */
	virtual Calibrator* clone() const = 0;
	virtual Calibrator* create() const = 0;

	int hasParticles() const {
		return myParticleDeposits.size();
	}
	
	std::vector<ParticleDepositPtr> getParticles() {
		return myParticleDeposits;
	}

	//	virtual void setDetectorElements(const std::vector<DetectorElement*> elements) {
	//		myDetectorElements = elements;
	//	}

protected:
	virtual std::map<DetectorElementPtr, double>
			getCalibrationCoefficientsCore() throw(PFToolsException&);

	//virtual DetectorElementPtr getOffsetElementCore();

	//static DetectorElement* offsetElement;
	std::vector<DetectorElementPtr> myDetectorElements;
	std::vector<ParticleDepositPtr> myParticleDeposits;
};

typedef boost::shared_ptr<Calibrator> CalibratorPtr;

}

#endif /*CALIBRATOR_HH_*/
