#include "RecoParticleFlow/PFClusterTools/interface/Exercises.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.h"
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/Operators.h"

#include "RecoParticleFlow/PFClusterTools/interface/SingleParticleWrapper.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationResultWrapper.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationProvenance.h"

#include <vector>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <functional>
#include <TH1F.h>
#include <TTree.h>
#include <TRandom2.h>

using namespace pftools;
using namespace std;

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
	//DetectorElementPtr offset(new DetectorElement(OFFSET, 1.0));
	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	std::vector<DetectorElementPtr> elements;
	//elements.push_back(offset);
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
	linCal->addDetectorElement(offset);
	linCal->addDetectorElement(ecal);
	linCal->addDetectorElement(hcal);

	sm->createCalibrators(*linCal, 2, 0, 2.5, 2, -3.15, 3.15, 1, 0, 1000);

	std::vector<DetectorElementPtr> elements;
	elements.push_back(offset);
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

		if (c == 0) {
			std::cout << "Couldn't find calibrator for particle?!\n";
			std::cout << "\t"<< *pd << "\n";
		} else {
			c->addParticleDeposit(pd);
		}
	}

	std::cout << "Producing Exercises.root..."<< std::endl;
	TFile output("Exercises.root", "recreate");

	std::map<SpaceVoxelPtr, CalibratorPtr>* detectorMap = sm->getCalibrators();
	std::map<SpaceVoxelPtr, TH1F> before;
	std::map<SpaceVoxelPtr, TH1F> after;
	std::cout << "Calling performance evaluation..."<< std::endl;
	//evaluatePerformance(detectorMap, before, after);
	std::cout << std::endl;
	std::cout << "Writing out histograms..."<< std::endl;
	writeOutHistos(before);
	writeOutHistos(after);
	std::cout << "Closing files..."<< std::endl;
	output.Write();
	//For some weird weird reason, closing the file causes a glibc seg fault! Mad.
	//output.Close();
	std::cout << "Finished exercise."<< std::endl;

}

void Exercises::evaluatePerformance(
		const std::map<SpaceVoxelPtr, CalibratorPtr>* const detectorMap,
		TFile& treeFile) const {

	//	//loop over detectorMap
	//	for (std ::map<SpaceVoxelPtr, CalibratorPtr>::const_iterator
	//			cit = detectorMap->begin(); cit != detectorMap->end(); ++cit) {
	//		SpaceVoxelPtr sv = (*cit).first;
	//		std::cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
	//		std::cout << *sv << "\n";
	//		CalibratorPtr c = (*cit).second;
	//		std::cout << "Calibrator has "<< c->hasParticles() << " particles\n";
	//
	//		//Global energy resolution improvement for this calibrator
	//		double oldReso(0.0), newReso(0.0);
	//
	//		if (c->hasParticles()) {
	//			std::vector<ParticleDepositPtr> csParticles = c->getParticles();
	//
	//			//define histogram names
	//			std::string svName;
	//			sv->getName(svName);
	//			std::string histoNamePre("hPre");
	//			std::string histoNamePost("hPost");
	//			histoNamePre.append(svName);
	//			histoNamePost.append(svName);
	//
	//			//Construct histograms
	//			TH1F pre(histoNamePre.c_str(), histoNamePre.c_str(), 100, 0, 200);
	//			TH1Fpost(histoNamePost.c_str(), histoNamePost.c_str(), 100, 0, 200);
	//
	//			for (std::vector<ParticleDepositPtr>::iterator
	//					it = csParticles.begin(); it!= csParticles.end(); ++it) {
	//				ParticleDepositPtr pd = *it;
	//				oldReso += pd->getEnergyResolution();
	//				pre.Fill(pd->getRecEnergy());
	//			}
	//			std::map<DetectorElementPtr, double>
	//					calibs = c->getCalibrationCoefficients();
	//			for (std::map<DetectorElementPtr, double>::iterator
	//					it = calibs.begin(); it != calibs.end(); ++it) {
	//				DetectorElementPtr de = (*it).first;
	//				de->setCalib((*it).second);
	//			}
	//			for (std::vector<ParticleDepositPtr>::iterator
	//					it = csParticles.begin(); it!= csParticles.end(); ++it) {
	//				ParticleDepositPtr pd = *it;
	//				newReso += pd->getEnergyResolution();
	//				post.Fill(pd->getRecEnergy());
	//			}
	//			std::cout << "*** Consistency Check ***\n";
	//			c->getCalibrationCoefficients();
	//			std::cout << "*** End of check ***\n";
	//			for (std::map<DetectorElementPtr, double>::iterator
	//					it = calibs.begin(); it != calibs.end(); ++it) {
	//				DetectorElementPtr de = (*it).first;
	//				de->setCalib(1.0);
	//			}
	//
	//			before[sv] = pre;
	//			after[sv] = post;
	//
	//			std::cout << "\tOld reso:\t"<< oldReso / csParticles.size() * 100.0
	//					<< "\n";
	//			std::cout << " \tNew reso:\t"<< newReso / csParticle s.size() *100.0
	//			<< "\n";
	//		}
	//		std::cout << "*** Completion of evaluation ***\n";
	//	}
}


void Exercises::calibrateParticlesAndWriteOut(
		TFile& f) const {
	cout << __PRETTY_FUNCTION__ << "\n";
	
	TreeUtility tu;
	const std::vector<ParticleDepositPtr> input = tu.extractParticles(f);
	
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
	
	cout << "Made detector elements.\n";
	
	
	sm->createCalibrators(*linCal, 1, -4.0, 4.0, 1, -3.15, 3.15, 1, 0, 1000);
	
	cout << "Initialised SpaceManager and calibrators.\n";
	
	std::vector<DetectorElementPtr> elements;
	elements.push_back(offset);
	elements.push_back(ecal);
	elements.push_back(hcal);

	//Initialise calibrators with particles
	int count(0);
	for (std::vector<ParticleDepositPtr>::const_iterator cit = input.begin(); cit
			!= input.end(); ++cit) {
		ParticleDepositPtr pd = *cit;
		CalibratorPtr c = sm->findCalibrator(pd->getEta(), pd->getPhi(),
				pd->getTruthEnergy());
		//std::cout << *pd << "\n";

		if (c == 0) {
			std::cout << "Couldn't find calibrator for particle?!\n";
			std::cout << "\t"<< *pd << "\n";
		} else {
			c->addParticleDeposit(pd);
		}
		if(count % 1000 == 0) {
			cout << count << "\n";
			cout << *pd << "\n";
			
		}
		++count;
	}
	
	cout << "Extracted particles.\n";
	f.Close();
	TFile treeFile("Exercises.root", "recreate");
	//Set up trees and such like...
	TTree tree("CalibratedParticles", "");
	SingleParticleWrapper* mySpw = new SingleParticleWrapper();
	tree.Branch("SingleParticleWrapper", "pftools::SingleParticleWrapper",
			&mySpw, 32000, 2);
	
	cout << "Initialised tree.\n";

	//calibrate
	std::map<SpaceVoxelPtr, CalibratorPtr>
			* smCalibrators = sm->getCalibrators();
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
		cout << *sv << "\n";
		
		CalibratorPtr c = (*it).second;
		cout << "Calibrator has " << c->hasParticles() << " particles.\n";
		
		//get results for each calibrator
		int k(0);
		if (c->hasParticles()) {
			std::map<DetectorElementPtr, double>
					calibs = c->getCalibrationCoefficients();

			std::vector<ParticleDepositPtr> csParticles = c->getParticles();
			for (std::vector<ParticleDepositPtr>::iterator
					pit = csParticles.begin(); pit != csParticles.end(); ++pit) {
				ParticleDepositPtr pd = *pit;
				mySpw->reset();
				mySpw->eEcal = pd->getRecEnergy(ecal);
				mySpw->eHcal = pd->getRecEnergy(hcal);
				mySpw->trueEnergy = pd->getTruthEnergy();

				CalibrationResultWrapper crwPre;
				crwPre.ecalEnergy_ = pd->getRecEnergy(ecal);
				crwPre.hcalEnergy_ = pd->getRecEnergy(hcal);
				crwPre.particleEnergy_ = pd->getRecEnergy();
				crwPre.provenance_ = UNCALIBRATED;
				crwPre.target_ = UNDEFINED;
				mySpw->calibrations_.push_back(crwPre);

				//evaluate calibration
				for (std::map<DetectorElementPtr, double>::iterator
						deit = calibs.begin(); deit != calibs.end(); ++deit) {
					DetectorElementPtr de = (*deit).first;
					de->setCalib((*deit).second);
				}

				CalibrationResultWrapper crwPos;
				crwPos.ecalEnergy_ = pd->getRecEnergy(ecal);
				crwPos.hcalEnergy_ = pd->getRecEnergy(hcal);
				crwPos.particleEnergy_ = pd->getRecEnergy();
				crwPos.provenance_ = LINEAR;
				crwPos.target_ = CLUSTER;

				mySpw->calibrations_.push_back(crwPos);

				tree.Fill();

				for (std::map<DetectorElementPtr, double>::iterator
						deit = calibs.begin(); deit != calibs.end(); ++deit) {
					DetectorElementPtr de = (*deit).first;
					de->setCalib(1.0);
				}
			}
		}
	}
	cout << "Writing output tree...\n";
	tree.Write();
	treeFile.Write();
	cout << "Done." << endl;
}

void Exercises::writeOutHistos(std::map<SpaceVoxelPtr, TH1F>& input) const {
	std::vector<TH1F> histos;
	valueVector(input, histos);
	std::for_each(histos.begin(), histos.end(), writeHisto);
}
