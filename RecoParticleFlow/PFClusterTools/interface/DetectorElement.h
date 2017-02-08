#ifndef DETECTORELEMENT_HH_
#define DETECTORELEMENT_HH_

#include <vector>
#include <boost/shared_ptr.hpp>
/*
 * This is a dirty macro that allows you to make vectors of DetectorElements within CINT
 */
//#ifdef __MAKECINT__
//#pragma link C++ class std::vector<minimiser::DetectorElement>
//#endif
//
//#ifdef __MAKECINT__
//#pragma link C++ class std::vector<minimiser::DetectorElement*>
//#endif

#include "RecoParticleFlow/PFClusterTools/interface/DetectorElementType.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"

#include "TH2F.h"
#include "TH1F.h"
#include <iosfwd>

namespace pftools {

/**
 \class DetectorElement 
 \brief Represents an energy-measuring region of our detector.
 
 The Calibrator class will make a calibration constant for each DetectorElement passed to it.

 \author Jamie Ballin
 \date   April 2008
 */
class DetectorElement {
public:

	DetectorElement(DetectorElementType type, double calib = 1.0);

	inline DetectorElementType getType() const {
		return myType;
	}
	;

	virtual ~DetectorElement();

	/*
	 * Returns a global detector element calibration.
	 */
	double getCalib() const {
		return getCalibCore();
	}

	/*
	 * Returns the calibration for this detector element as a function
	 * of eta and phi.
	 */
	double getCalib(double eta, double phi) const {
		return getCalibCore(eta, phi);
	}

	/*
	 * Set the calibration of this detector element. Must be > 0.
	 */
	void setCalib(double calib) throw(PFToolsException&) {
		setCalibCore(calib);
	}

	friend std::ostream& operator<<(std::ostream& s, const DetectorElement& de);


private:
	virtual double getCalibCore() const;
	virtual double getCalibCore(double eta, double phi) const;
	virtual void setCalibCore(double calib) throw(PFToolsException&);
	
	DetectorElement(const DetectorElement& de);
	DetectorElementType myType;
	double myCalib;

};

typedef boost::shared_ptr<DetectorElement> DetectorElementPtr;

std::ostream& operator<<(std::ostream& s, const DetectorElement& de);

}

#endif /*DETECTORELEMENT_HH_*/
