#ifndef CALIBRATOR_HH_
#define CALIBRATOR_HH_

#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.hh"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.hh"
#include <map>
#include <vector>

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

	virtual void addDetectorElement(DetectorElement* const de);

	virtual void addParticleDeposit(ParticleDeposit* pd);

	/*
	 * Returns the calibration coefficient for each detector element, using data
	 * from all particle depositions stored within.
	 */
	virtual std::map<DetectorElement*, double> getCalibrationCoefficients() throw(
			PFToolsException&);

	virtual DetectorElement* getOffsetElement();
	
	/* 
	 * Here we use the virtual constructor idea to allow for plug-and-play Calibrators
	 * See http://www.parashift.com/c++-faq-lite/virtual-functions.html#faq-20.8
	 */
	virtual Calibrator* clone() const = 0;
	virtual Calibrator* create() const = 0;
	
	int hasParticles() const {
		return myParticleDeposits.size();
	}
	
//	virtual void setDetectorElements(const std::vector<DetectorElement*> elements) {
//		myDetectorElements = elements;
//	}

protected:

	DetectorElement offsetElement;
	std::vector<DetectorElement*> myDetectorElements;
	std::vector<ParticleDeposit*> myParticleDeposits;
};
}

#endif /*CALIBRATOR_HH_*/
