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

#include "RecoParticleFlow/PFClusterTools/interface/DetectorElementType.hh"
#include "RecoParticleFlow/PFClusterTools/interface/MinimiserException.hh"

#include "TH2F.h"
#include "TH1F.h"

namespace pftools {

class DetectorElement {
public:
	
	typedef boost::shared_ptr<DetectorElement> DetectorElementPtr;
	
	DetectorElement(DetectorElementType type, double calib = 1.0);

	inline DetectorElementType getType() const {
		return myType;
	}
	;

	virtual ~DetectorElement();

	/*
	 * Returns a global detector element calibration.
	 */
	virtual double getCalib() const;

	/*
	 * Returns the calibration for this detector element as a function
	 * of eta and phi.
	 */
	virtual double getCalib(double eta, double phi) const;

	/*
	 * Set the calibration of this detector element. Must be > 0.
	 */
	void setCalib(double calib) throw(MinimiserException&);
	

	//friend std::ostream& operator<<(std::ostream& s, const DetectorElement& de);

	friend std::ostream& operator<<(std::ostream& s,
			const DetectorElement& de);

private:
	DetectorElement(const DetectorElement& de);
	DetectorElementType myType;
	double myCalib;
	

	/*
	 * For general ROOT dictionary building happiness!
	 */
	std::vector<DetectorElement*> deps_;
	//std::vector<DetectorElement> des_;
};
}

#endif /*DETECTORELEMENT_HH_*/
