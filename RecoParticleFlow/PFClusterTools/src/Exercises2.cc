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
#include <TH3F.h>
#include <TF2.h>
#include <TH1F.h>
#include <vector>

using namespace pftools;

void resetElements(DetectorElementPtr de) {
	de->setCalib(1.0);
}

Exercises2::~Exercises2() {
}

Exercises2::Exercises2(double lowE, double highE, unsigned divE, double lowEta,
		double highEta, double divEta, double lowPhi, double highPhi,
		unsigned divPhi, bool withOffset) :
	lowE_(lowE), highE_(highE), divE_(divE), lowEta_(lowEta),
			highEta_(highEta), divEta_(divEta), lowPhi_(lowPhi),
			highPhi_(highPhi), divPhi_(divPhi), withOffset_(withOffset), target_(CLUSTER) {
	
}

void Exercises2::gaussianFits(TFile& exercisefile,
		std::vector<Calibratable>& calibs) {
	std::cout << __PRETTY_FUNCTION__ << "\n";
	exercisefile.cd("/");
	exercisefile.mkdir("gaussianFits");
	exercisefile.cd("/gaussianFits");
	unsigned events(calibs.size());
	// How many bins? (Rather ad-hoc!)
	// take a safety margin of a factor of 4 over statistical fluctuations (sqrt(N))
	// bin width = 4 sqrt(N)
	// N bins = N/(4 sqrt(N)) = sqrt(N) / 4
	//unsigned nbins = static_cast<unsigned>(ceil(sqrt(events) / 4.0));
	unsigned nbins = static_cast<unsigned>((highE_ / 2.0- lowE_));
	std::cout << "nbins in x, y = "<< nbins << ", z = "<< nbins * 4<< "\n";
	TH3F energyPlane("energyPlane", "Energy plane;ECAL;HCAL;True", nbins,
			lowE_, highE_/2.0, nbins, lowE_, highE_/2.0, nbins * 4, lowE_,
			highE_);

	for (std::vector<Calibratable>::iterator it = calibs.begin(); it
			!= calibs.end(); ++it) {
		Calibratable c = *it;
		energyPlane.Fill(c.cluster_meanEcal_.energy_,
				c.cluster_meanHcal_.energy_, c.sim_energyEvent_);

	}
	std::cout << "Fitting slices...\n";
	energyPlane.FitSlicesZ();
	TH2D* energyPlane_1 = (TH2D*) gDirectory->FindObject("energyPlane_1");
	TF2* f2 = new TF2("f2","[0]+[1]*x+[2]*y",2,20,2,20);
	std::cout << "Fitting plane...\n";
	energyPlane_1->Fit("f2");
	std::cout << "p0, p1, p2 = a, b, c respectively.\n";
	energyPlane.Write();
	energyPlane_1->Write();
	std::cout << "Done gaussian fits.\n";
}

void Exercises2::calibrateCalibratables(const std::string& sourcefile,
		const std::string& exercisefile) {

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
	tu.convertCalibratablesToParticleDeposits(calibVec, pdVec, target_, offset,
			ecal, hcal, withOffset_);
	source->Close();
	std::cout << "Closed source file. Opening exercises file...\n";
	TFile* exercises = new TFile(exercisefile.c_str(), "recreate");
	TH1F droppedParticles("droppedParticles", "droppedParticles", 100000, 0,
			100000);
	std::cout << "Particle deposit vec has "<< pdVec.size() << " entries\n";

	//calibrate
	boost::shared_ptr<Calibrator> linCal(new LinearCalibrator());
	boost::shared_ptr<Calibrator> hcalCal(new LinearCalibrator());
	boost::shared_ptr<Calibrator> ecalCal(new LinearCalibrator());
	//Tell the calibrator which detector elements should be calibrated
	if (withOffset_) {
		linCal->addDetectorElement(offset);
		hcalCal->addDetectorElement(offset);
		ecalCal->addDetectorElement(offset);
	}
	linCal->addDetectorElement(ecal);
	linCal->addDetectorElement(hcal);

	//ecalCal->addDetectorElement(hcal);
	//hcalCal->addDetectorElement(ecal);
	//hcalCal->addDetectorElement(offset);
	hcalCal->addDetectorElement(hcal);
	ecalCal->addDetectorElement(ecal);
	
	boost::shared_ptr<SpaceManager> sm(new SpaceManager());
	sm->createCalibrators(*linCal, divEta_, lowEta_, highEta_, divPhi_,
			lowPhi_, highPhi_, divE_, lowE_, highE_);
	boost::shared_ptr<SpaceManager> esm(new SpaceManager());
	esm->createCalibrators(*ecalCal, divEta_, lowEta_, highEta_, divPhi_,
			lowPhi_, highPhi_, divE_, lowE_, highE_);
	boost::shared_ptr<SpaceManager> hsm(new SpaceManager());
	hsm->createCalibrators(*hcalCal, divEta_, lowEta_, highEta_, divPhi_,
			lowPhi_, highPhi_, divE_, lowE_, highE_);

	std::cout << "Initialised SpaceManager and calibrators.\n";

	std::vector<DetectorElementPtr> elements;
	if (withOffset_)
		elements.push_back(offset);
	elements.push_back(ecal);
	elements.push_back(hcal);

	//Make a subset of calibrators for ecal and hcal only depositions
	//boost::shared_ptr<Calibrator> ecalCal(new LinearCalibrator());
	//ecalCal->addDetectorElement(ecal)

	//Initialise calibrators with particles
	int count(0);
	int dropped(0);
	double eCut(0.5);
	for (std::vector<ParticleDepositPtr>::const_iterator cit = pdVec.begin(); cit
			!= pdVec.end(); ++cit) {
		ParticleDepositPtr pd = *cit;
		if (count%100== 0)
			std::cout << *pd;
		if (pd->getRecEnergy(ecal) > eCut && pd->getRecEnergy(hcal) > eCut) {
			CalibratorPtr c = sm->findCalibrator(pd->getEta(), pd->getPhi(),
					pd->getTruthEnergy());
			//std::cout << *pd << "\n";
			if (c == 0) {
				std::cout << "Couldn't find calibrator for particle?!\n";
				std::cout << "\t"<< *pd << "\n";
			} else {
				c->addParticleDeposit(pd);
			}
		} else if (pd->getRecEnergy(ecal) < eCut && pd->getRecEnergy(hcal) > eCut) {
			CalibratorPtr c = hsm->findCalibrator(pd->getEta(), pd->getPhi(),
					pd->getTruthEnergy());
			//std::cout << *pd << "\n";
			if (c == 0) {
				std::cout << "Couldn't find calibrator for particle?!\n";
				std::cout << "\t"<< *pd << "\n";
			} else {
				c->addParticleDeposit(pd);
			}
			//std::cout << "Dropping deposit: \n" << *pd;
		} else if (pd->getRecEnergy(hcal) < eCut && pd->getRecEnergy(ecal) > eCut) {
			CalibratorPtr c = esm->findCalibrator(pd->getEta(), pd->getPhi(),
					pd->getTruthEnergy());
			//std::cout << *pd << "\n";
			if (c == 0) {
				std::cout << "Couldn't find calibrator for particle?!\n";
				std::cout << "\t"<< *pd << "\n";
			} else {
				c->addParticleDeposit(pd);
			}
			//std::cout << "Dropping deposit: \n" << *pd;
		} else {
			++dropped;
			droppedParticles.Fill(count);
		}

		++count;
	}
	std::cout << "Dropped "<< dropped << " particles.\n";
	//calibrate
	std::map<SpaceVoxelPtr, CalibratorPtr>
			* smCalibrators = sm->getCalibrators();
	std::map<SpaceVoxelPtr, CalibratorPtr>
			* hsmCalibrators = hsm->getCalibrators();
	std::map<SpaceVoxelPtr, CalibratorPtr>
				* esmCalibrators = esm->getCalibrators();

	TTree tree("CalibratedParticles", "");
	Calibratable* calibrated = new Calibratable();
	tree.Branch("Calibratable", "pftools::Calibratable", &calibrated, 32000, 2);

	std::cout << "Initialised tree.\n";
	std::cout << "*** Performance for ECAL + HCAL calibration ***\n";
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		std::cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
		std::cout << *sv << "\n";

		CalibratorPtr c = (*it).second;

		evaluateCalibrator(c, tree, calibrated, ecal, hcal, LINEAR);
		std::for_each(elements.begin(), elements.end(), resetElements);
	}
	std::for_each(elements.begin(), elements.end(), resetElements);
	std::cout << "*** Performace of HCAL ONLY calibration ***\n";
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = hsmCalibrators->begin(); it != hsmCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		std::cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
		std::cout << *sv << "\n";

		CalibratorPtr c = (*it).second;

		evaluateCalibrator(c, tree, calibrated, ecal, hcal, LINEAR);
		std::for_each(elements.begin(), elements.end(), resetElements);
	}
	std::for_each(elements.begin(), elements.end(), resetElements);
	std::cout << "*** Performace of ECAL ONLY calibration ***\n";
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = esmCalibrators->begin(); it != esmCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		std::cout << "*** Performance evalutation for SpaceVoxel ***\n\t";
		std::cout << *sv << "\n";

		CalibratorPtr c = (*it).second;

		evaluateCalibrator(c, tree, calibrated, ecal, hcal, LINEAR);
		std::for_each(elements.begin(), elements.end(), resetElements);
	}

	//save results
	std::cout << "Writing output tree...\n";
	tree.Write();
	droppedParticles.Write();
	gaussianFits(*exercises, calibVec);
	exercises->Write();
	exercises->Close();
	std::cout << "Done."<< std::endl;

}

void Exercises2::evaluateCalibrator(CalibratorPtr c, TTree& tree,
		Calibratable* calibrated, DetectorElementPtr ecal,
		DetectorElementPtr hcal, CalibrationProvenance cp) {
	//get results for each calibrator
	std::cout << "Calibrator has "<< c->hasParticles() << " particles.\n";
	if (c->hasParticles() > 50) {
		std::map<DetectorElementPtr, double>
				calibs = c->getCalibrationCoefficients();

		std::vector<ParticleDepositPtr> csParticles = c->getParticles();
		for (std::vector<ParticleDepositPtr>::iterator
				pit = csParticles.begin(); pit!= csParticles.end(); ++pit) {
			ParticleDepositPtr pd = *pit;
			calibrated->reset();
			calibrated->rechits_meanEcal_.energy_ = pd->getRecEnergy(ecal);
			calibrated->rechits_meanHcal_.energy_ = pd->getRecEnergy(hcal);
			calibrated->sim_energyEvent_ = pd->getTruthEnergy();
			calibrated->sim_etaEcal_ = pd->getEta();
			
			for (std::map<DetectorElementPtr, double>::iterator
					deit = calibs.begin(); deit != calibs.end(); ++deit) {
				DetectorElementPtr de = (*deit).first;
				de->setCalib(1.0);
			}

			CalibrationResultWrapper crwPre;
			crwPre.ecalEnergy_ = pd->getRecEnergy(ecal);
			crwPre.hcalEnergy_ = pd->getRecEnergy(hcal);
			crwPre.particleEnergy_ = pd->getRecEnergy();
			crwPre.truthEnergy_ = pd->getTruthEnergy();
			crwPre.provenance_ = UNCALIBRATED;
			crwPre.target_ = target_;
			crwPre.bias_ = crwPre.bias();
			crwPre.targetFuncContrib_ = pd->getTargetFunctionContrib();
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
			crwPos.provenance_ = cp;
			crwPos.bias_ = crwPos.bias();
			crwPos.targetFuncContrib_ = pd->getTargetFunctionContrib();
			crwPos.target_ = target_;

			calibrated->calibrations_.push_back(crwPos);

			tree.Fill();


		}
	} else {
		std::cout
				<< "WARNING: Calibrator had less than 51 particles; skipping."
				<< std::endl;
	}
}

void Exercises2::doPlots(const std::string& sourcefile,
		std::vector<CalibrationTarget>& targets) {

}
