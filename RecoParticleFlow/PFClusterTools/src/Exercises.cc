#include "RecoParticleFlow/PFClusterTools/interface/Exercises.hh"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.hh"
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.hh"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.hh"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.hh"
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.hh"
#include "RecoParticleFlow/PFClusterTools/interface/Operators.h"

#include <vector>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <functional>
#include <TH1F.h>

#include <TRandom2.h>

using namespace pftools;



void writeHisto(TH1F& h) {
	h.Write();
}

Exercises::Exercises() {
}

Exercises::~Exercises() {
}

void Exercises::testTreeUtility(TFile& f) const {
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

void Exercises::testCalibrators() const {
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

void Exercises::testCalibrationFromTree(TFile& f) const {

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

	sm->createCalibrators(*linCal, 2, 0, 2.5, 2, -3.15, 3.15, 1, 0, 1000);

	std::vector<DetectorElementPtr> elements;
	//elements.push_back(offset);
	elements.push_back(ecal);
	elements.push_back(hcal);

	std::vector<ParticleDepositPtr> particles;

	TreeUtility tu;
	tu.recreateFromRootFile(f, elements, particles);
	std::vector<std::string> results;
	for (std::vector<ParticleDepositPtr>::iterator it = particles.begin(); it
			!= particles.end(); ++it) {
		ParticleDepositPtr pd = *it;

		CalibratorPtr c = sm->findCalibrator(pd->getEta(), pd->getPhi(),
				pd->getTruthEnergy());
		//std::cout << *pd << "\n";

		if (c == 0){
			std::cout << "Couldn't find calibrator for particle?!\n";
			std::cout << "\t" << *pd << "\n";
		}
		else {
			c->addParticleDeposit(pd);
		}
	}
	
	std::cout << "Producing Exercises.root..." << std::endl;
	TFile output("Exercises.root", "recreate");
	std::map<SpaceVoxelPtr, CalibratorPtr>* detectorMap = sm->getCalibrators();
	std::map<SpaceVoxelPtr, TH1F> before;
	std::map<SpaceVoxelPtr, TH1F> after;
	std::cout << "Calling performance evaluation..." << std::endl;
	evaluatePerformance(detectorMap, before, after);
	std::cout << std::endl;
	std::cout << "Writing out histograms..." << std::endl;
	writeOutHistos(before);
	writeOutHistos(after);
	std::cout << "Closing files..." << std::endl;
	output.Write();
	//For some weird weird reason, closing the file causes a glibc seg fault! Mad.
	//output.Close();
	std::cout << "Finished exercise." << std::endl;

}

void Exercises::evaluatePerformance(
		const std::map<SpaceVoxelPtr, CalibratorPtr>* const detectorMap,
		std::map<SpaceVoxelPtr, TH1F>& before,
		std::map<SpaceVoxelPtr, TH1F>& after) const {

	//loop over detectorMap
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::const_iterator
			cit = detectorMap->begin(); cit != detectorMap->end(); ++cit) {
		SpaceVoxelPtr sv = (*cit).first;
		std::cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
		std::cout << *sv << "\n";
		CalibratorPtr c = (*cit).second;
		std::cout << "Calibrator has "<< c->hasParticles() << " particles\n";

		//Global energy resolution improvement for this calibrator
		double oldReso(0.0), newReso(0.0);

		if (c->hasParticles()) {
			std::vector<ParticleDepositPtr> csParticles = c->getParticles();

			//define histogram names
			std::string svName;
			sv->getName(svName);
			std::string histoNamePre("hPre");
			std::string histoNamePost("hPost");
			histoNamePre.append(svName);
			histoNamePost.append(svName);

			//Construct histograms
			TH1F pre(histoNamePre.c_str(), histoNamePre.c_str(), 100, 0, 200);
			TH1F
					post(histoNamePost.c_str(), histoNamePost.c_str(), 100, 0,
							200);

			for (std::vector<ParticleDepositPtr>::iterator
					it = csParticles.begin(); it!= csParticles.end(); ++it) {
				ParticleDepositPtr pd = *it;
				oldReso += pd->getEnergyResolution();
				pre.Fill(pd->getRecEnergy());
			}
			std::map<DetectorElementPtr, double>
					calibs = c->getCalibrationCoefficients();
			for (std::map<DetectorElementPtr, double>::iterator
					it = calibs.begin(); it != calibs.end(); ++it) {
				DetectorElementPtr de = (*it).first;
				de->setCalib((*it).second);
			}
			for (std::vector<ParticleDepositPtr>::iterator
					it = csParticles.begin(); it!= csParticles.end(); ++it) {
				ParticleDepositPtr pd = *it;
				newReso += pd->getEnergyResolution();
				post.Fill(pd->getRecEnergy());
			}
			std::cout << "*** Consistency Check ***\n";
			c->getCalibrationCoefficients();
			std::cout << "*** End of check ***\n";
			for (std::map<DetectorElementPtr, double>::iterator
					it = calibs.begin(); it != calibs.end(); ++it) {
				DetectorElementPtr de = (*it).first;
				de->setCalib(1.0);
			}

			before[sv] = pre;
			after[sv] = post;

			std::cout << "\tOld reso:\t"<< oldReso / csParticles.size() * 100.0
					<< "\n";
			std::cout << "\tNew reso:\t"<< newReso / csParticles.size() * 100.0
					<< "\n";
		}
		std::cout << "*** Completion of evaluation ***\n";
	}

}

void Exercises::calibrateAndRewriteParticles(TFile& f) const {
	
}

void Exercises::writeOutHistos(std::map<SpaceVoxelPtr, TH1F>& input) const {
	std::vector<TH1F> histos;
	valueVector(input, histos);
	std::for_each(histos.begin(), histos.end(),  writeHisto);
}
