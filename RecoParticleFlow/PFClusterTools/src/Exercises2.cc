#include "RecoParticleFlow/PFClusterTools/interface/Exercises2.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"
#include <TFile.h>
#include <TTree.h>
#include <vector>
using namespace pftools;

Exercises2::Exercises2() {
}

Exercises2::~Exercises2() {
}

void Exercises2::calibrateCalibratables(const std::string& sourcefile, const std::string& exercisefile) {

	std::cout << "Welcome to "<< __PRETTY_FUNCTION__ << "\n";
	std::cout << "Opening TFile...\n";
	//open tfile
	TFile* source = new TFile(sourcefile.c_str());
	if (source == 0) {
		std::string desc("Couldn't open file ");
		desc += sourcefile;
		PFToolsException e(desc);
		throw e;
	}

	std::cout << "Extracting calibratables...\n";
	//use tree utility to extract calibratables
	TreeUtility tu;
	std::vector<Calibratable> calibVec;
	tu.getCalibratablesFromRootFile(*source, calibVec);

	std::cout << "Got a vector of calibratables of size "<< calibVec.size()
			<< "\n";
	//initialise detector elements
	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	DetectorElementPtr offset(new DetectorElement(OFFSET, 1.0));

	//convert calibratables to particle deposits
	std::vector<ParticleDepositPtr> pdVec;
	tu.convertCalibratablesToParticleDeposits(calibVec, pdVec, RECHIT, offset, ecal,
			hcal);

	std::cout << "Particle deposit vec has "<< pdVec.size() << " entries\n";

	//calibrate

	boost::shared_ptr<Calibrator> linCal(new LinearCalibrator());

	//Tell the calibrator which detector elements should be calibrated
	linCal->addDetectorElement(offset);
	linCal->addDetectorElement(ecal);
	linCal->addDetectorElement(hcal);
	boost::shared_ptr<SpaceManager> sm(new SpaceManager());
	sm->createCalibrators(*linCal, 1, 0.0, 4.0, 1, -3.15, 3.15, 1, 0, 50);

	std::cout << "Initialised SpaceManager and calibrators.\n";

	std::vector<DetectorElementPtr> elements;
	elements.push_back(offset);
	elements.push_back(ecal);
	elements.push_back(hcal);

	//Initialise calibrators with particles
	int count(0);
	for (std::vector<ParticleDepositPtr>::const_iterator cit = pdVec.begin(); cit
			!= pdVec.end(); ++cit) {
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

		++count;
	}

	//calibrate
	std::map<SpaceVoxelPtr, CalibratorPtr>
			* smCalibrators = sm->getCalibrators();
	source->Close();
	std::cout << "Closed source file. Opening exercises file...\n";
	TFile* exercises = new TFile(exercisefile.c_str(), "recreate");
	TTree tree("CalibratedParticles", "");
	Calibratable* calibrated = new Calibratable();
	tree.Branch("Calibratable", "pftools::Calibratable", &calibrated, 32000, 2);

	std::cout << "Initialised tree.\n";

	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		std::cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
		std::cout << *sv << "\n";

		CalibratorPtr c = (*it).second;
		std::cout << "Calibrator has "<< c->hasParticles() << " particles.\n";

		//get results for each calibrator
		int k(0);
		if (c->hasParticles()) {
			std::map<DetectorElementPtr, double>
					calibs = c->getCalibrationCoefficients();

			std::vector<ParticleDepositPtr> csParticles = c->getParticles();
			for (std::vector<ParticleDepositPtr>::iterator
					pit = csParticles.begin(); pit != csParticles.end(); ++pit) {
				ParticleDepositPtr pd = *pit;
				calibrated->reset();
				calibrated->rechits_energyEcal_ = pd->getRecEnergy(ecal);
				calibrated->rechits_energyHcal_ = pd->getRecEnergy(hcal);
				calibrated->sim_energyEvent_ = pd->getTruthEnergy();

				CalibrationResultWrapper crwPre;
				crwPre.ecalEnergy_ = pd->getRecEnergy(ecal);
				crwPre.hcalEnergy_ = pd->getRecEnergy(hcal);
				crwPre.particleEnergy_ = pd->getRecEnergy();
				crwPre.truthEnergy_ = pd->getTruthEnergy();
				crwPre.provenance_ = UNCALIBRATED;
				crwPre.target_ = RECHIT;
				crwPre.bias_ = crwPre.bias();
				calibrated->calibrations_.push_back(crwPre);

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
				crwPos.truthEnergy_ = pd->getTruthEnergy();
				crwPos.provenance_ = LINEAR;
				crwPos.bias_ = crwPos.bias();
				crwPos.target_ = RECHIT;

				calibrated->calibrations_.push_back(crwPos);

				tree.Fill();

				for (std::map<DetectorElementPtr, double>::iterator
						deit = calibs.begin(); deit != calibs.end(); ++deit) {
					DetectorElementPtr de = (*deit).first;
					de->setCalib(1.0);
				}
			}
		}
	}
	//save results
	std::cout << "Writing output tree...\n";
	tree.Write();
	exercises->Write();
	exercises->Close();
	std::cout << "Done."<< std::endl;

	
}

void Exercises2::doPlots(const std::string& sourcefile, std::vector<CalibrationTarget>& targets) {
	
}
