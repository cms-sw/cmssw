#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.hh"
using namespace minimiser;

Calibrator::Calibrator() : offsetElement(OFFSET) {
}

Calibrator::~Calibrator() {
}

void Calibrator::addDetectorElement(DetectorElement* const de) {
	//std::cout << "myDetecotElements has size: " << myDetectorElements.size() << "before addition.\n";
	myDetectorElements.push_back(de);
}
void Calibrator::addParticleDeposit(ParticleDeposit* pd) {
	myParticleDeposits.push_back(pd);
}

std::map<DetectorElement*, double> Calibrator::getCalibrationCoefficients() throw(
		MinimiserException&) {

	std::cout << __PRETTY_FUNCTION__
			<< ": Not implemented in default Calibrator class!\n";
	std::cout << "\tWARNING: returning empty map.\n";
	std::map<DetectorElement*, double> answers;
	return answers;
}

DetectorElement* Calibrator::getOffsetElement()  {
	DetectorElement* el = &offsetElement;
	return el;
}
