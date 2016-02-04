#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"
#include <ostream>
using namespace pftools;
//A comment
DetectorElement::DetectorElement(DetectorElementType type, double calib) :
	myType(type), myCalib(calib) {

}

void DetectorElement::setCalibCore(double calib) throw(PFToolsException&){
	//I'll tolerate very small negative numbers (artefacts of the minimisation algo
	//but otherwise this shouldn't be allowed.
//	if(calib > -0.01) {
		myCalib = calib;
//	}
//	else {
//		MinimiserException me("Setting calibration <= 0!");
//		throw me;
//	}
}

double DetectorElement::getCalibCore() const {
	if(myType == OFFSET && myCalib == 1) {
		return 1.0;
	}
	return myCalib;
}

double DetectorElement::getCalibCore(double eta, double phi) const {
	return getCalib();
}

DetectorElement::~DetectorElement() {
}


std::ostream& pftools::operator<<(std::ostream& s, const DetectorElement& de) {
	s << "DetectorElement: " << pftools::DetElNames[de.getType()] << ", \tcalib: " << de.getCalib();

	return s;
}

