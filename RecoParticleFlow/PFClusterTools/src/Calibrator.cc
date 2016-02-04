#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
using namespace pftools;

//DetectorElement* Calibrator::offsetElement = new DetectorElement(OFFSET);

Calibrator::Calibrator(){

}

Calibrator::~Calibrator() {
}

void Calibrator::addDetectorElement(DetectorElementPtr const de) {
	//std::cout << "myDetecotElements has size: " << myDetectorElements.size() << "before addition.\n";
	myDetectorElements.push_back(de);
}
void Calibrator::addParticleDeposit(ParticleDepositPtr pd) {
	myParticleDeposits.push_back(pd);
}

std::map<DetectorElementPtr, double> Calibrator::getCalibrationCoefficientsCore() throw(
		PFToolsException&) {

	std::cout << __PRETTY_FUNCTION__
			<< ": Not implemented in default Calibrator class!\n";
	std::cout << "\tWARNING: returning empty map.\n";
	std::map<DetectorElementPtr, double> answers;
	return answers;
}

//DetectorElementPtr Calibrator::getOffsetElementCore()  {
//	DetectorElementPtr el(offsetElement);
//	return el;
//}
