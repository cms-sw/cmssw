#include "RecoParticleFlow/PFClusterTools/interface/Deposition.h"
using namespace pftools;
Deposition::Deposition(DetectorElementPtr element, double eta, double phi,
		double energy, double depth) :
	myElementPtr(element), myEta(eta), myPhi(phi), myEnergy(energy),
			myDepth(depth) {
}

Deposition::~Deposition() {
}

std::ostream& pftools::operator<<(std::ostream& s, const Deposition& d) {
	s << "Deposit's type: "<< *(d.getDetectorElement()) << "\tE:\t"<< d.getEnergy()
			<< ", \teta:\t"<< d.getEta() << ", \tmyPhi:\t"<< d.getPhi() << "\n";
	return s;
}
