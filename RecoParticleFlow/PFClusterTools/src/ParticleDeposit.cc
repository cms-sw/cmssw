#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElementType.h"
using namespace pftools;

unsigned ParticleDeposit::count = 0;

ParticleDeposit::ParticleDeposit(double truthEnergy, double eta, double phi) :
	myId(count), myTruthEnergy(truthEnergy), myEta(eta), myPhi(phi) {
	++count;
}

ParticleDeposit::~ParticleDeposit() {
}

void ParticleDeposit::addRecDeposition(const Deposition& rec) {
	myRecDepositions.push_back(rec);
}

void ParticleDeposit::addTruthDeposition(const Deposition& truth) {
	myTruthDepositions.push_back(truth);
}

std::vector<Deposition> ParticleDeposit::getTruthDepositions() const {
	return myTruthDepositions;
}

const std::vector<Deposition>& ParticleDeposit::getRecDepositions() const {
	return myRecDepositions;
}

double ParticleDeposit::getRecEnergy(const DetectorElementPtr de) const {
	double energy(0);
	for (std::vector<Deposition>::const_iterator cit = myRecDepositions.begin(); cit
			!= myRecDepositions.end(); ++cit) {
		Deposition d = *cit;

		if (d.getDetectorElement()->getType() == de->getType()) {
			energy += de->getCalib(d.getEta(), d.getPhi()) * d.getEnergy();
		}

	}
	return energy;
}

void ParticleDeposit::setRecEnergy(const DetectorElementPtr de, double energy) {
	for (std::vector<Deposition>::const_iterator cit = myRecDepositions.begin(); cit
			!= myRecDepositions.end(); ++cit) {
		Deposition d = *cit;

		if (d.getDetectorElement()->getType() == de->getType()) {
			d.setEnergy(energy);
		}

	}
}

double ParticleDeposit::getTruthEnergy(const DetectorElementPtr de) const {
	double energy(0);
	for (std::vector<Deposition>::const_iterator
			cit = myTruthDepositions.begin(); cit!= myTruthDepositions.end(); ++cit) {
		Deposition d = *cit;
		if (d.getDetectorElement() == de) {
			energy += d.getEnergy();
		}
	}
	assert(!(energy > 0));
	return energy;

}

double ParticleDeposit::getRecEnergy() const {
	double energy(0);
	for (std::vector<Deposition>::const_iterator cit = myRecDepositions.begin(); cit
			!= myRecDepositions.end(); ++cit) {
		Deposition d = *cit;
//		if (d.getDetectorElement()->getType() == OFFSET && d.getDetectorElement()->getCalib() == 1.0) {
//			//don't add a tiny amount!
//		} else {
			energy += d.getDetectorElement()->getCalib(d.getEta(), d.getPhi()) * d.getEnergy();
//		}
	}

	//assert(!(energy < 0));
	return energy;

}

double ParticleDeposit::getEnergyResolution() const {
	//assert(!(getRecEnergy() / myTruthEnergy < 0.0));
	return fabs((getRecEnergy() - myTruthEnergy) / sqrt(myTruthEnergy));
}

double ParticleDeposit::getTargetFunctionContrib() const {
	//assert(!(getRecEnergy() / myTruthEnergy < 0.0));
	return pow((getRecEnergy() - myTruthEnergy), 2);
}

std::ostream& pftools::operator<<(std::ostream& s, const pftools::ParticleDeposit& p) {
	s << "Particle id:\t" << p.getId() << ", \t trueEnergy: " << p.getTruthEnergy() << "\n";
	s.width(3);
	s << "\tEta:\t" << p.getEta() << ",\tphi:\t" << p.getPhi() << "\n";
	for (std::vector<Deposition>::const_iterator cit = p.getRecDepositions().begin(); cit
			!= p.getRecDepositions().end(); ++cit) {
		Deposition d = *cit;
		DetectorElementPtr de(d.getDetectorElement());
		s << "\t" << *de << ": \t=> E_contrib = ";
		s << de->getCalib(d.getEta(), d.getPhi()) * d.getEnergy() << "\n";
	}
	s << "\tTotalRecEnergy: " << p.getRecEnergy() << ",\t res: " << p.getEnergyResolution() * 100 << "%\n";
	return s;
}
