#include "RecoParticleFlow/PFClusterTools/interface/SpaceVoxel.h"
#include <iostream>
#include "RecoParticleFlow/PFClusterTools/interface/ToString.h"
using namespace pftools;

SpaceVoxel::SpaceVoxel(double etaBegin, double etaEnd, double phiBegin,
		double phiEnd, double energyBegin, double energyEnd) :
	myEtaMin(etaBegin), myEtaMax(etaEnd), myPhiMin(phiBegin), myPhiMax(phiEnd),
			myEnergyMin(energyBegin), myEnergyMax(energyEnd) {

}

SpaceVoxel::~SpaceVoxel() {
}

bool SpaceVoxel::contains(const double& eta, const double& phi, const double& energy) const {
	if(containsEta(eta) && containsPhi(phi) && containsEnergy(energy))
		return true;
	return false;
}

bool SpaceVoxel::containsEta(const double& eta) const {
	if(myEtaMin == myEtaMax)
		return true;
	if(eta < myEtaMax && eta >= myEtaMin)
			return true;
		return false;
}

bool SpaceVoxel::containsPhi(const double& phi) const {
	if(myPhiMin == myPhiMax)
		return true;
	if(phi < myPhiMax && phi >= myPhiMin)
			return true;
		return false;
	
}

bool SpaceVoxel::containsEnergy(const double& energy) const {
	if(myEnergyMin == myEnergyMax)
		return true;
	if(energy < myEnergyMax && energy >= myEnergyMin)
		return true;
	return false;
}

void SpaceVoxel::print(std::ostream& s) const {
	s << "SpaceVoxel: eta: [" << myEtaMin << ", " << myEtaMax << "]\t phi: [" << myPhiMin << ". " << myPhiMax << "]\t energy: [" << myEnergyMin << ", " << myEnergyMax << "]";
}

void SpaceVoxel::getName(std::string& s) const {
	s.append("SpaceVoxel: eta: [");
	s.append(toString(myEtaMin));
	s.append(", ");
	s.append(toString(myEtaMax));
	s.append("] phi: [");
	s.append(toString(myPhiMin));
	s.append(", ");
	s.append(toString(myPhiMax));
	s.append("], en: [");
	s.append(toString(myEnergyMin));
	s.append(", ");
	s.append(toString(myEnergyMax));
	s.append("]");
}

std::ostream& pftools::operator<<(std::ostream& s, const pftools::SpaceVoxel& sv) {
	sv.print(s);
	return s;
}
