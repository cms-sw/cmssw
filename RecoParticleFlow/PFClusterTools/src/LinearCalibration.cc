#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibration.h"

using namespace pftools;

LinearCalibration::LinearCalibration() :
	a_(0), b_(0), c_(0) {
}

LinearCalibration::LinearCalibration(CalibrationTarget t, double a, double b,
		double c) :
	 a_(a), b_(b), c_(c) {
	targ_ = t;
	prov_ = LINEAR;
}

LinearCalibration::LinearCalibration(CalibrationTarget t, double b, double c) :
	 a_(-1.0), b_(b), c_(c) {
	targ_ = t;
	prov_ = LINEAR;

}

LinearCalibration::~LinearCalibration() {
}

void LinearCalibration::calibrateCore(const CalibrationResultWrapper& crwInput,
		CalibrationResultWrapper& crwOutput) {
	if (targ_ == UNDEFINED) {
		std::cout << __PRETTY_FUNCTION__ << ": WARNING! Target is undefined!\n";
		return;
	} else {
		crwOutput.reset();
		crwOutput.target_ = Calibration::targ_;
		crwOutput.provenance_ = LINEAR;

		crwOutput.ecalEnergy_ = b_ * crwInput.ecalEnergy_;
		crwOutput.hcalEnergy_ = c_ * crwInput.hcalEnergy_;
		crwOutput.particleEnergy_ = a_ + crwOutput.ecalEnergy_
				+ crwOutput.hcalEnergy_;
		crwOutput.truthEnergy_ = crwInput.truthEnergy_;
		crwOutput.compute();

	}

}

std::ostream& pftools::operator<<(std::ostream& s, const LinearCalibration& lc) {
	s << "LinearCalibration:  a, b, c = {" << lc.a_ << ", " << lc.b_ << ", " << lc.c_ << "}\n";
	return s;
}
