#include "RecoParticleFlow/PFClusterTools/interface/Exercises3.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/Region.h"
#include "RecoParticleFlow/PFClusterTools/interface/IO.h"

#include <TFile.h>
#include <TTree.h>
#include <TH3F.h>
#include <TF2.h>
#include <TH1F.h>
#include <TGraph.h>
#include <TF1.h>
#include <vector>
#include <TROOT.h>

using namespace pftools;

void resetElement3(DetectorElementPtr de) {
	de->setCalib(1.0);
}

Exercises3::~Exercises3() {
	calibResultsFile_.close();
}

Exercises3::Exercises3(IO* options) :
	withOffset_(false), target_(CLUSTER), threshold_(30), options_(options),
			debug_(0) {

	options_->GetOpt("exercises", "withOffset", withOffset_);
	options_->GetOpt("exercises", "threshold", threshold_);
	options_->GetOpt("exercises", "debug", debug_);

	/* Initialise PFClusterCalibration appropriately. */
	double g0, g1, e0, e1;
	options_->GetOpt("correction", "globalP0", g0);
	options_->GetOpt("correction", "globalP1", g1);
	options_->GetOpt("correction", "lowEP0", e0);
	options_->GetOpt("correction", "lowEP1", e1);
	clusterCalibration_.setCorrections(e0, e1, g0, g1);

	double ecalECut, hcalECut;
	options_->GetOpt("evolution", "ecalECut", ecalECut);
	options_->GetOpt("evolution", "hcalECut", hcalECut);
	clusterCalibration_.setEcalHcalEnergyCuts(ecalECut, hcalECut);

	int allowNegative(0);
	options_->GetOpt("correction", "allowNegativeEnergy", allowNegative);
	clusterCalibration_.setAllowNegativeEnergy(allowNegative);

	int doCorrection(1);
	options_->GetOpt("correction", "doCorrection", doCorrection);
	clusterCalibration_.setDoCorrection(doCorrection);

	int doEtaCorrection(0);
	options_->GetOpt("correction", "doEtaCorrection", doEtaCorrection);
	clusterCalibration_.setDoEtaCorrection(doEtaCorrection);

	double barrelEta;
	options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEta);
	clusterCalibration_.setBarrelBoundary(barrelEta);

	double maxEToCorrect(100.0);
	options_->GetOpt("correction", "maxEToCorrect", maxEToCorrect);
	clusterCalibration_.setMaxEToCorrect(maxEToCorrect);

	std::vector<std::string>* names = clusterCalibration_.getKnownSectorNames();
	for (std::vector<std::string>::iterator i = names->begin(); i
			!= names->end(); ++i) {
		std::string sector = *i;
		std::vector<double> params;
		options_->GetOpt("evolution", sector.c_str(), params);
		clusterCalibration_.setEvolutionParameters(sector, params);
	}

	std::vector<double> etaParams;
	options_->GetOpt("evolution", "etaCorrection", etaParams);
	clusterCalibration_.setEtaCorrectionParameters(etaParams);

	std::cout << clusterCalibration_ << "\n";

	std::string outputFileName;
	options_->GetOpt("results", "calibParamOutput", outputFileName);
	calibResultsFile_.open(outputFileName.c_str());
	calibResultsFile_ << "//Hello from your friendly PFClusterTools!\n";
	if (debug_ > 0)
		std::cout << __PRETTY_FUNCTION__ << ": finished.\n";

}

void Exercises3::calibrateCalibratables(TChain& sourceTree,
		const std::string& exercisefile) {

	if (debug_ > 0) {
		std::cout << "Welcome to "<< __PRETTY_FUNCTION__ << "\n";
		std::cout << "Opening TTree...\n";
	}
//	//open tfile
//	TFile* source = new TFile(sourcefile.c_str());
//	if (source == 0) {
//		std::string desc("Couldn't open file ");
//		desc += sourcefile;
//		PFToolsException e(desc);
//		throw e;
//	}
//	if (debug_ > 0)
//		std::cout << "Extracting calibratables...\n";
	//use tree utility to extract calibratables
	TreeUtility tu;
	std::vector<Calibratable> calibVec;

	tu.getCalibratablesFromRootFile(sourceTree, calibVec);
	if (debug_ > 0)
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
	//source->Close();

	TFile* exercises = new TFile(exercisefile.c_str(), "recreate");
	TH1F droppedParticles("droppedParticles", "droppedParticles", 100000, 0,
			100000);
	if (debug_ > 0)
		std::cout << "Particle deposit vec has "<< pdVec.size() << " entries\n";

	//calibrate
	if (debug_ > 1)
		std::cout << "Creating calibrator clones and space managers\n";
	boost::shared_ptr<Calibrator> linCal(new LinearCalibrator());

	//Tell the calibrator which detector elements should be calibrated
	if (withOffset_) {
		linCal->addDetectorElement(offset);

	}
	linCal->addDetectorElement(ecal);
	linCal->addDetectorElement(hcal);

	double barrelEta;
	options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEta);
	boost::shared_ptr<SpaceManager> sm(new SpaceManager("ecalAndHcal"));
	sm->setBarrelLimit(barrelEta);
	sm->createCalibrators(*linCal);

	if (debug_ > 1)
		std::cout << "Initialised SpaceManager and calibrators.\n";
	elements_.clear();
	if (withOffset_)
		elements_.push_back(offset);
	elements_.push_back(ecal);
	elements_.push_back(hcal);

	//Initialise calibrators with particles
	int dropped(0);

	double eCut(0.0);
	double hCut(0.0);
	options_->GetOpt("evolution", "ecalECut", eCut);
	options_->GetOpt("evolution", "hcalECut", hCut);
	if (debug_ > 0)
		std::cout << "Using a ECAL and HCAL energy cuts of "<< eCut << " and "
				<< hCut << " GeV\n";
	if (debug_ > 1)
		std::cout << "Assigning particles to space managers and calibrators.\n";

	//This is just a convenience plot to check on the hcal
	for (std::vector<ParticleDepositPtr>::const_iterator cit = pdVec.begin(); cit
			!= pdVec.end(); ++cit) {
		ParticleDepositPtr pd = *cit;
		CalibratorPtr c = sm->findCalibrator(pd->getEta(), pd->getPhi(),
				pd->getTruthEnergy());
		//std::cout << *pd << "\n";
		if (c == 0) {
			std::cout << "Couldn't find calibrator for particle?!\n";
			std::cout << "\t"<< *pd << "\n";

			dropped++;
		} else {
			c->addParticleDeposit(pd);
		}

	}

	if (debug_ > 1)
		std::cout << "Dropped "<< dropped << " particles.\n";

	/* Done assignments, now calibrate */
	if (debug_ > 1)
		std::cout
				<< "Assignments complete, starting calibration and analysis.\n";

	//calibrate
	std::map<SpaceVoxelPtr, CalibratorPtr> * smCalibrators =
			sm->getCalibrators();

	TTree tree("CalibratedParticles", "");
	Calibratable* calibrated = new Calibratable();
	tree.Branch("Calibratable", "pftools::Calibratable", &calibrated, 32000, 2);
	if (debug_ > 1)
		std::cout << "Initialised tree.\n";

	/* ECAL and HCAL */
	std::cout << "*** Performance for ECAL + HCAL calibration ***\n";
	getCalibrations(sm);
	exercises->cd("/");
	exercises->mkdir("ecalAndHcal");
	exercises->cd("/ecalAndHcal");
	evaluateSpaceManager(sm, elements_);
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator it =
			smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		CalibratorPtr c = (*it).second;
		std::for_each(elements_.begin(), elements_.end(), resetElement3);
		evaluateCalibrator(sm, c, tree, calibrated, ecal, hcal, offset, LINEAR,
				LINEARCORR);

		std::for_each(elements_.begin(), elements_.end(), resetElement3);
	}
	sm->printCalibrations(std::cout);

	exercises->cd("/");

	//save results
	std::cout << "Writing output tree...\n";
	tree.Write();
	droppedParticles.Write();
	//gaussianFits(*exercises, calibVec);
	exercises->Write();
	exercises->Close();
	std::cout << "Done."<< std::endl;

}

void Exercises3::getCalibrations(SpaceManagerPtr s) {

	std::map<SpaceVoxelPtr, CalibratorPtr>* smCalibrators = s->getCalibrators();

	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator it =
			smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		CalibratorPtr c= (*it).second;
		std::for_each(elements_.begin(), elements_.end(), resetElement3);
		if (c->hasParticles() > static_cast<int>(threshold_)) {
			std::map<DetectorElementPtr, double> calibs =
					c->getCalibrationCoefficients();
			s->assignCalibration(c, calibs);
		}
	}
}

void Exercises3::evaluateSpaceManager(SpaceManagerPtr s,
		const std::vector<DetectorElementPtr>& detEls) {

	int autoFit(0);
	options_->GetOpt("evolution", "autoFit", autoFit);
	std::cout << "AutoFit option = "<< autoFit << "\n";

	std::vector<double> ecalBarrel;
	std::vector<double> ecalEndcap;
	std::vector<double> hcalBarrel;
	std::vector<double> hcalEndcap;

	double minE, maxE;
	options_->GetOpt("evolution", "evolutionFunctionMinE", minE);
	options_->GetOpt("evolution", "evolutionFunctionMaxE", maxE);

	int basePlots(0);
	options_->GetOpt("evolution", "basePlots", basePlots);

	int useTruth(1);
	options_->GetOpt("evolution", "basePlotsUseTruth", useTruth);

	if (debug_ > 1&& basePlots > 0)
		std::cout << "Option to generate evolution plots invoked.\n";

	if (autoFit == 0) {
		std::cout << "Fixing parameters for evolution functions.\n";

		options_->GetOpt("evolution", "ecalHcalEcalBarrel", ecalBarrel);
		options_->GetOpt("evolution", "ecalHcalEcalEndcap", ecalEndcap);
		options_->GetOpt("evolution", "ecalHcalHcalBarrel", hcalBarrel);
		options_->GetOpt("evolution", "ecalHcalHcalEndcap", hcalEndcap);
		assert(ecalBarrel.size() == 6 && ecalEndcap.size() == 6);
		assert(hcalBarrel.size() == 6 && hcalEndcap.size() == 6);

		for (std::vector<DetectorElementPtr>::const_iterator i = detEls.begin(); i
				!= detEls.end(); ++i) {
			DetectorElementPtr d = *i;
			std::cout << "Fixing evolution for "<< *d << "\n";
			int mode(0);
			options_->GetOpt("spaceManager", "interpolationMode", mode);

			std::string name("Func_");
			name.append(DetElNames[d->getType()]);
			name.append("_");

			/* Fitting for barrel */
			std::string barrelName(name);
			barrelName.append(RegionNames[BARREL_POS]);
			std::cout << "\tFixing "<< RegionNames[BARREL_POS]<< "\n";
			TF1
					fBarrel(barrelName.c_str(),
							"([0]*[5]*x)*([5]*x<[1])+([2]+[3]*exp([4]*[5]*x))*([5]*x>[1])");

			if (d->getType() == ECAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator dit =
						ecalBarrel.begin(); dit!= ecalBarrel.end(); ++dit) {
					fBarrel.FixParameter(count, *dit);
					++count;
				}

			}
			if (d->getType() == HCAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator dit =
						hcalBarrel.begin(); dit!= hcalBarrel.end(); ++dit) {
					fBarrel.FixParameter(count, *dit);
					++count;
				}

			}
			if (useTruth)
				fBarrel.FixParameter(5, 1.0);

			fBarrel.SetMinimum(0);
			s->addEvolution(d, BARREL_POS, fBarrel);

			if (basePlots > 0) {
				TH1* slices = s->extractEvolution(d, BARREL_POS, fBarrel,
						useTruth);
				slices->Write();
			}
			fBarrel.Write();

			/* Fitting for endcap */
			std::string endcapName(name);
			endcapName.append(RegionNames[ENDCAP_POS]);
			std::cout << "\tFixing "<< RegionNames[ENDCAP_POS]<< "\n";
			TF1
					fEndcap(endcapName.c_str(),
							"([0]*[5]*x)*([5]*x<[1])+([2]+[3]*exp([4]*[5]*x))*([5]*x>[1])");
			

			if (d->getType() == ECAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator dit =
						ecalEndcap.begin(); dit!= ecalEndcap.end(); ++dit) {
					fEndcap.FixParameter(count, *dit);
					++count;
				}

			}
			if (d->getType() == HCAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator dit =
						hcalEndcap.begin(); dit!= hcalEndcap.end(); ++dit) {
					fEndcap.FixParameter(count, *dit);
					++count;
				}

			}
			if (useTruth)
				fEndcap.FixParameter(5, 1.0);

			fEndcap.SetMinimum(0);
			s->addEvolution(d, ENDCAP_POS, fEndcap);
			if (basePlots > 0) {
				TH1* slices = s->extractEvolution(d, ENDCAP_POS, fEndcap,
						useTruth);
				slices->Write();
			}
			fEndcap.Write();
		}

	}

}

void Exercises3::evaluateCalibrator(SpaceManagerPtr s, CalibratorPtr c,
		TTree& tree, Calibratable* calibrated, DetectorElementPtr ecal,
		DetectorElementPtr hcal, DetectorElementPtr offset,
		CalibrationProvenance cp, CalibrationProvenance cpCorr) {

	if (c->hasParticles() > static_cast<int>(threshold_)) {
		std::map<DetectorElementPtr, double> calibs = s->getCalibration(c);

		std::vector<ParticleDepositPtr> csParticles = c->getParticles();
		unsigned count(0);
		int mode(0);
		options_->GetOpt("spaceManager", "interpolationMode", mode);
		if (debug_ > 1) {
			std::cout << "Using interpolation mode " << mode << "\n";
		}

		for (std::vector<ParticleDepositPtr>::iterator zit = csParticles.begin(); zit
				!= csParticles.end(); ++zit) {
			ParticleDepositPtr pd = *zit;
			calibrated->reset();
			calibrated->rechits_meanEcal_.energy_ = pd->getRecEnergy(ecal);
			calibrated->rechits_meanHcal_.energy_ = pd->getRecEnergy(hcal);
			calibrated->sim_energyEvent_ = pd->getTruthEnergy();
			calibrated->sim_etaEcal_ = pd->getEta();

			for (std::map<DetectorElementPtr, double>::iterator deit =
					calibs.begin(); deit != calibs.end(); ++deit) {
				DetectorElementPtr de = (*deit).first;
				de->setCalib(1.0);
			}

			CalibrationResultWrapper crwPre;
			crwPre.ecalEnergy_ = pd->getRecEnergy(ecal);
			crwPre.hcalEnergy_ = pd->getRecEnergy(hcal);
			crwPre.particleEnergy_ = pd->getRecEnergy();
			crwPre.truthEnergy_ = pd->getTruthEnergy();
			crwPre.provenance_ = UNCALIBRATED;
			crwPre.targetFuncContrib_ = pd->getTargetFunctionContrib();
			crwPre.target_ = target_;
			crwPre.compute();
			calibrated->calibrations_.push_back(crwPre);

			double tempEnergy = pd->getRecEnergy();
			//evaluate calibration
			for (std::map<DetectorElementPtr, double>::iterator deit =
					calibs.begin(); deit != calibs.end(); ++deit) {
				DetectorElementPtr de = (*deit).first;

				if (mode == 1)
					de->setCalib(s->interpolateCoefficient(de,
							pd->getTruthEnergy(), pd->getEta(), pd->getPhi()));
				else if (mode == 2|| mode == 3|| mode == 4)
					de->setCalib(s->evolveCoefficient(de, tempEnergy,
							pd->getEta(), pd->getPhi()));
				else
					de->setCalib((*deit).second);
			}
			if (debug_ > 8) {
				std::cout << "POST ECAL HCAL coeff: " << ecal->getCalib() << ", " << hcal->getCalib() << "\n";
			}

			CalibrationResultWrapper crwPos;
			crwPos.ecalEnergy_ = pd->getRecEnergy(ecal);
			crwPos.hcalEnergy_ = pd->getRecEnergy(hcal);
			crwPos.b_ = ecal->getCalib();
			crwPos.c_ = hcal->getCalib();
			crwPos.particleEnergy_ = pd->getRecEnergy();
			crwPos.truthEnergy_ = pd->getTruthEnergy();
			crwPos.provenance_ = cp;
			crwPos.compute();

			crwPos.targetFuncContrib_ = pd->getTargetFunctionContrib();
			crwPos.target_ = target_;
			calibrated->calibrations_.push_back(crwPos);

			//same again, but applying correction
			if (cpCorr != NONE) {
				CalibrationResultWrapper crwCorr;

				crwCorr.ecalEnergy_
						= clusterCalibration_.getCalibratedEcalEnergy(
								crwPre.ecalEnergy_, crwPre.hcalEnergy_,
								pd->getEta(), pd->getPhi());
				crwCorr.hcalEnergy_
						= clusterCalibration_.getCalibratedHcalEnergy(
								crwPre.ecalEnergy_, crwPre.hcalEnergy_,
								pd->getEta(), pd->getPhi());
				if (debug_ > 8) {
					if(crwPre.ecalEnergy_  > 0 && crwPre.hcalEnergy_ >0)
					std::cout << "CORR ECAL HCAL coeff: " << crwCorr.ecalEnergy_ / crwPre.ecalEnergy_  << ", " << crwCorr.hcalEnergy_/ crwPre.hcalEnergy_ << "\n\n";
				}

				crwCorr.particleEnergy_
						= clusterCalibration_.getCalibratedEnergy(
								crwPre.ecalEnergy_, crwPre.hcalEnergy_,
								pd->getEta(), pd->getPhi());

				crwCorr.b_ = ecal->getCalib();
				crwCorr.c_ = hcal->getCalib();

				crwCorr.truthEnergy_ = pd->getTruthEnergy();
				crwCorr.provenance_ = cpCorr;
				crwCorr.targetFuncContrib_ = pd->getTargetFunctionContrib();
				crwCorr.target_ = target_;
				crwCorr.compute();
				calibrated->calibrations_.push_back(crwCorr);

				crwPos.targetFuncContrib_ = pd->getTargetFunctionContrib();
				crwPos.target_ = target_;
				calibrated->calibrations_.push_back(crwPos);
			}
			tree.Fill();
			++count;

		}
	}
}

