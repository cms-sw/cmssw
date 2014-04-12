#include "RecoParticleFlow/PFClusterTools/interface/SpaceVoxel.h"
#include <iostream>
#include "RecoParticleFlow/PFClusterTools/interface/ToString.h"
using namespace pftools;

SpaceVoxel::SpaceVoxel(double etaBegin, double etaEnd, double phiBegin,
		double phiEnd, double energyBegin, double energyEnd, bool ecalValid, bool hcalValid) :
	myEtaMin(etaBegin), myEtaMax(etaEnd), myPhiMin(phiBegin), myPhiMax(phiEnd),
			myEnergyMin(energyBegin), myEnergyMax(energyEnd),
			ecalValid_(ecalValid), hcalValid_(hcalValid) {
	if (!ecalValid_ && !hcalValid_) {
		//erm, it has to be valid for one of them at least!
		std::cout << __PRETTY_FUNCTION__
				<< ": WARNING! Constructed with both ecalValid and hcalValid = false!"
				<< std::endl;
	}
}

SpaceVoxel::~SpaceVoxel() {
}

bool SpaceVoxel::contains(const double& eta, const double& phi,
		const double& energy) const {
	if (containsEta(eta) && containsPhi(phi) && containsEnergy(energy))
		return true;
	return false;
}

bool SpaceVoxel::contains(const double& eta, const double& phi,
		const double& energy, const bool& ecalValid, const bool& hcalValid) const {
	if (contains(eta, phi, energy) && ecalValid == ecalValid_ && hcalValid
			== hcalValid_)
		return true;
	return false;
}

bool SpaceVoxel::containsEta(const double& eta) const {
	if (myEtaMin == myEtaMax)
		return true;
	if (eta < myEtaMax && eta >= myEtaMin)
		return true;
	//::cout << "\teta fails!\n";
	return false;
}

bool SpaceVoxel::containsPhi(const double& phi) const {
	if (myPhiMin == myPhiMax)
		return true;
	if (phi < myPhiMax && phi >= myPhiMin)
		return true;
	//std::cout << "\tphi fails!\n";
	return false;

}

bool SpaceVoxel::containsEnergy(const double& energy) const {
	if (myEnergyMin == myEnergyMax)
		return true;
	if (energy < myEnergyMax && energy >= myEnergyMin)
		return true;
	//std::cout << "\tenergy fails!: input " << energy << " not in " << myEnergyMin << ", " << myEnergyMax <<"\n";
	return false;
}

void SpaceVoxel::print(std::ostream& s) const {
	s << "SpaceVoxel: ";
	if (ecalValid_)
		s << "E";
	if (hcalValid_)
		s << "H, ";
	s << "eta: ["<< myEtaMin << ", "<< myEtaMax << "]\t phi: ["<< myPhiMin
			<< ". "<< myPhiMax << "]\t energy: ["<< myEnergyMin<< ", "
			<< myEnergyMax << "]";
}

void SpaceVoxel::getName(std::string& s) const {
	s.append("SpaceVoxel: ");
	if (ecalValid_)
		s.append("E");
	if (hcalValid_)
		s.append("H");
	s.append(", eta: [");
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

bool SpaceVoxel::operator()(const SpaceVoxel& sv1, const SpaceVoxel& sv2)  {
	if(sv1.minEnergy() < sv2.maxEnergy())
		return true;
	
	return false;
}

bool SpaceVoxel::operator()(const SpaceVoxelPtr& svp1, const SpaceVoxelPtr& svp2)  {
	SpaceVoxel sv1 = *svp1;
	SpaceVoxel sv2 = *svp2;
	if(sv1.minEnergy() < sv2.maxEnergy())
			return true;
		
		return false;
}

std::ostream& pftools::operator<<(std::ostream& s, const pftools::SpaceVoxel& sv) {
	sv.print(s);
	return s;
}
