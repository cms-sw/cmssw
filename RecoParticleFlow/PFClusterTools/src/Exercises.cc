#include "RecoParticleFlow/PFClusterTools/interface/Exercises.hh"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.hh"
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.hh"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.hh"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.hh"
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.hh"

#include <vector>
#include <boost/shared_ptr.hpp>
#include <iostream>

#include <TRandom2.h>

using namespace pftools;

Exercises::Exercises() {
}

Exercises::~Exercises() {
}

void Exercises::testTreeUtility(TFile& f) {
	std::cout << __PRETTY_FUNCTION__ << "\n";
	std::cout << "Starting tests...\n";
	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	std::vector<DetectorElementPtr> elements;
	elements.push_back(ecal);
	elements.push_back(hcal);
	std::cout << "Made detector elements...\n";
	std::cout << "Recreating from root file...\n";
	std::vector<ParticleDepositPtr> particles;
	TreeUtility tu;
	tu.recreateFromRootFile(f, elements, particles);
	std::cout << "Finished tests.\n";

}

void Exercises::testCalibrators() {
	std::cout << __PRETTY_FUNCTION__ << "\n";
	std::cout << "Wilkommen und ze testCalibrator funktionnen!\n";
	std::cout << "Starting tests...\n";

	boost::shared_ptr<SpaceManager> sm(new SpaceManager());
	//Make detector elements to start with
	DetectorElementPtr offset(new DetectorElement(OFFSET, 1.0));

	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	//std::cout << "Made: " << *ecalB;
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	//	/std::cout << "Made: " << *hcalB;
	boost::shared_ptr<Calibrator> linCal(new LinearCalibrator());

	//Tell the calibrator which detector elements should be calibrated
	linCal->addDetectorElement(offset);
	linCal->addDetectorElement(ecal);
	linCal->addDetectorElement(hcal);

	sm->createCalibrators(*linCal, 1, -10.0, 10.0, 4, 0, 360, 1, 0, 800);

	std::vector<DetectorElementPtr> elements;
	elements.push_back(offset);
	elements.push_back(ecal);
	elements.push_back(hcal);

	TRandom2 rand;
	std::vector<ParticleDepositPtr > particles;
	for (unsigned u(0); u < 10000; ++u) {
		double eta, phi, energy, ecalFrac;

		eta = rand.Uniform(-10.0, 10.0);
		phi = rand.Uniform(0, 360);
		//energy = 100.0;
		energy = rand.Uniform(10.0, 190.0);
		ParticleDepositPtr pd(new ParticleDeposit(energy, eta, phi));
		ecalFrac = rand.Uniform(0, 1.0);
		Deposition dOffset(offset, eta, phi, 1.0);
		if (phi > 180)
			energy *= 2.0;
		Deposition dE(ecal, eta, phi, 0.8 * energy * ecalFrac - 0.2);
		Deposition dH(hcal, eta, phi, 1.6 * energy * (1 - ecalFrac)- 0.3);
		pd->addRecDeposition(dOffset);
		pd->addRecDeposition(dE);
		pd->addRecDeposition(dH);

		pd->addTruthDeposition(dOffset);
		pd->addTruthDeposition(dE);
		pd->addTruthDeposition(dH);

		particles.push_back(pd);

		CalibratorPtr c = sm->findCalibrator(eta, phi, energy);
		//std::cout << *pd << "\n";

		if (c == 0)
			std::cout << "Couldn't find calibrator for particle?!\n";
		else {
			c->addParticleDeposit(pd);
		}
	}

	std::map<SpaceVoxelPtr, CalibratorPtr>* detectorMap = sm->getCalibrators();
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::const_iterator
			cit = detectorMap->begin(); cit != detectorMap->end(); ++cit) {
		SpaceVoxelPtr sv = (*cit).first;
		std::cout << *sv << "\n";
		CalibratorPtr c = (*cit).second;
		std::cout << "Calibrator has "<< c->hasParticles() << " particles\n";
		if (c->hasParticles()) {
			std::map<DetectorElementPtr, double>
					calibs = c->getCalibrationCoefficients();
		}
	}
	std::cout << "Finished tests.\n";
}

void Exercises::testCalibrationFromTree(TFile& f) {

	boost::shared_ptr<SpaceManager> sm(new SpaceManager());
	//Make detector elements to start with
	DetectorElementPtr offset(new DetectorElement(OFFSET, 1.0));

	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	//std::cout << "Made: " << *ecalB;
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	//	/std::cout << "Made: " << *hcalB;
	boost::shared_ptr<Calibrator> linCal(new LinearCalibrator());

	//Tell the calibrator which detector elements should be calibrated
	//linCal->addDetectorElement(offset);
	linCal->addDetectorElement(ecal);
	linCal->addDetectorElement(hcal);

	sm->createCalibrators(*linCal, 1, -10.0, 10.0, 4, 0, 360, 1, 0, 800);

	std::vector<DetectorElementPtr> elements;
	//elements.push_back(offset);
	elements.push_back(ecal);
	elements.push_back(hcal);

	std::vector<ParticleDepositPtr> particles;

	TreeUtility tu;
	tu.recreateFromRootFile(f, elements, particles);

	for (std::vector<ParticleDepositPtr>::iterator it = particles.begin(); it
			!= particles.end(); ++it) {
		ParticleDepositPtr pd = *it;

		CalibratorPtr c = sm->findCalibrator(pd->getEta(), pd->getPhi(),
				pd->getTruthEnergy());
		//std::cout << *pd << "\n";

		if (c == 0)
			std::cout << "Couldn't find calibrator for particle?!\n";
		else {
			c->addParticleDeposit(pd);
		}
	}

	std::map<SpaceVoxelPtr, CalibratorPtr>* detectorMap = sm->getCalibrators();
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::const_iterator
			cit = detectorMap->begin(); cit != detectorMap->end(); ++cit) {
		SpaceVoxelPtr sv = (*cit).first;
		std::cout << *sv << "\n";
		CalibratorPtr c = (*cit).second;
		std::cout << "Calibrator has "<< c->hasParticles() << " particles\n";
		if (c->hasParticles()) {
			std::map<DetectorElementPtr, double>
					calibs = c->getCalibrationCoefficients();
		}
	}
}
