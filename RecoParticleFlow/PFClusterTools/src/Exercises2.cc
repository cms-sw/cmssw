#include "RecoParticleFlow/PFClusterTools/interface/Exercises2.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "RecoParticleFlow/PFClusterTools/interface/Region.h"

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

void resetElement(DetectorElementPtr de) {
	de->setCalib(1.0);
}

Exercises2::~Exercises2() {
	calibResultsFile_.close();
}

Exercises2::Exercises2(IO* options) :
	withOffset_(false), target_(CLUSTER), threshold_(30),
		 options_(options), debug_(0) {

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
	
	int allowNegative(0);
	options_->GetOpt("correction", "allowNegativeEnergy", allowNegative);
	clusterCalibration_.setAllowNegativeEnergy(allowNegative);
	
	int doCorrection(1);
	options_->GetOpt("correction", "doCorrection", doCorrection);
	clusterCalibration_.setDoCorrection(doCorrection);
	
	double barrelEta;
	options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEta);
	clusterCalibration_.setBarrelBoundary(barrelEta);
	
	double maxEToCorrect(100.0);
	options_->GetOpt("correction", "maxEToCorrect", maxEToCorrect);
	clusterCalibration_.setMaxEToCorrect(maxEToCorrect);
	
	std::vector<std::string>* names = clusterCalibration_.getKnownSectorNames();
	for(std::vector<std::string>::iterator i = names->begin(); i != names->end(); ++i) {
		std::string sector = *i;
		std::vector<double> params;
		options_->GetOpt("evolution", sector.c_str(), params);
		clusterCalibration_.setEvolutionParameters(sector, params);
	}
	
	int doEtaCorrection(1);
	options_->GetOpt("evolution", "doEtaCorrection", doEtaCorrection);
	clusterCalibration_.setDoEtaCorrection(doEtaCorrection);
	
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


void Exercises2::calibrateCalibratables(const std::string& sourcefile,
		const std::string& exercisefile) {

	if (debug_ > 0) {
		std::cout << "Welcome to "<< __PRETTY_FUNCTION__ << "\n";
		std::cout << "Opening TFile...\n";
	}
	//open tfile
	TFile* source = new TFile(sourcefile.c_str());
	if (source == 0) {
		std::string desc("Couldn't open file ");
		desc += sourcefile;
		PFToolsException e(desc);
		throw e;
	}
	if (debug_ > 0)
		std::cout << "Extracting calibratables...\n";
	//use tree utility to extract calibratables
	TreeUtility tu;
	std::vector<Calibratable> calibVec;
	PFToolsException e("TreeUtility has moved on, fix this up!");
	throw e;
	//tu.getCalibratablesFromRootFile(*source, calibVec);
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
	source->Close();
	if (debug_ > 0)
		std::cout << "Closed source file. Opening exercises file...\n";
	TFile* exercises = new TFile(exercisefile.c_str(), "recreate");
	TH1F droppedParticles("droppedParticles", "droppedParticles", 100000, 0,
			100000);
	if (debug_ > 0)
		std::cout << "Particle deposit vec has "<< pdVec.size() << " entries\n";

	//calibrate
	if (debug_ > 1)
		std::cout << "Creating calibrator clones and space managers\n";
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

	hcalCal->addDetectorElement(hcal);
	ecalCal->addDetectorElement(ecal);

	double barrelEta;
	options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEta);
	boost::shared_ptr<SpaceManager> sm(new SpaceManager("ecalAndHcal"));
	sm->setBarrelLimit(barrelEta);
	sm->createCalibrators(*linCal);
	boost::shared_ptr<SpaceManager> esm(new SpaceManager("ecalOnly"));
	esm->createCalibrators(*ecalCal);
	esm->setBarrelLimit(barrelEta);
	boost::shared_ptr<SpaceManager> hsm(new SpaceManager("hcalOnly"));
	hsm->createCalibrators(*hcalCal);
	hsm->setBarrelLimit(barrelEta);

	if (debug_ > 1)
		std::cout << "Initialised SpaceManager and calibrators.\n";
	elements_.clear();
	if (withOffset_)
		elements_.push_back(offset);
	elements_.push_back(ecal);
	elements_.push_back(hcal);

	//Initialise calibrators with particles
	int count(0);
	int dropped(0);

	double eCut(0.3);
	double hCut(0.5);
	options_->GetOpt("evolution", "ecalECut", eCut);
	options_->GetOpt("evolution", "hcalECut", hCut);
	if (debug_ > 0)
		std::cout << "Using a ECAL MIP cut of "<< eCut << " GeV\n";
	if (debug_ > 1)
		std::cout << "Assigning particles to space managers and calibrators.\n";

	//This is just a convenience plot to check on the hcal
	TH2F hcalOnlyInput("hcalOnlyInput", "hcalOnlyInput", 30, 0, 3, 50, 0, 5);
	for (std::vector<ParticleDepositPtr>::const_iterator cit = pdVec.begin(); cit
			!= pdVec.end(); ++cit) {
		ParticleDepositPtr pd = *cit;
		//		if (count%1000== 0)
		//			std::cout << *pd;

		if (pd->getRecEnergy(ecal) > eCut && pd->getRecEnergy(hcal) > hCut) {
			CalibratorPtr c = sm->findCalibrator(pd->getEta(), pd->getPhi(),
					pd->getTruthEnergy());
			//std::cout << *pd << "\n";
			if (c == 0) {
				if (debug_ > 1) {
					std::cout << "Couldn't find calibrator for particle?!\n";
					std::cout << "\t"<< *pd << "\n";
				}
				dropped++;
			} else {
				c->addParticleDeposit(pd);
			}
			/* HCAL fulfillment */
		} else if (pd->getRecEnergy(ecal) < eCut && pd->getRecEnergy(hcal)
				> hCut) {
			CalibratorPtr c = hsm->findCalibrator(pd->getEta(), pd->getPhi(),
					pd->getTruthEnergy());
			if (pd->getTruthEnergy() < 3.0) {
				hcalOnlyInput.Fill(pd->getTruthEnergy(), pd->getRecEnergy(hcal));
				//std::cout << *pd << "\n";
			}

			if (c == 0) {
				if (debug_ > 1) {
					std::cout << "Couldn't find calibrator for particle?!\n";
					std::cout << "\t"<< *pd << "\n";
				}
				dropped++;
			} else {
				pd->setRecEnergy(ecal, 0.0);
				c->addParticleDeposit(pd);
			}
			//std::cout << "Dropping deposit: \n" << *pd;
			/* ECAL fulfillment */
		} else if (pd->getRecEnergy(hcal) < hCut && pd->getRecEnergy(ecal)
				> eCut) {
			CalibratorPtr c = esm->findCalibrator(pd->getEta(), pd->getPhi(),
					pd->getTruthEnergy());

			//std::cout << *pd << "\n";
			if (c == 0) {
				if (debug_ > 1) {
					std::cout << "Couldn't find calibrator for particle?!\n";
					std::cout << "\t"<< *pd << "\n";
				}
				dropped++;
			} else {
				pd->setRecEnergy(hcal, 0.0);
				c->addParticleDeposit(pd);
			}
			//std::cout << "Dropping deposit: \n" << *pd;
		} else {
			++dropped;
			droppedParticles.Fill(count);
		}

		++count;
	}

	hcalOnlyInput.Write();

	if (debug_ > 1)
		std::cout << "Dropped "<< dropped << " particles.\n";

	/* Done assignments, now calibrate */
	if (debug_ > 1)
		std::cout
				<< "Assignments complete, starting calibration and analysis.\n";

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
	if (debug_ > 1)
		std::cout << "Initialised tree.\n";

	/* ECAL and HCAL */
	std::cout << "*** Performance for ECAL + HCAL calibration ***\n";
	getCalibrations(sm);
	exercises->cd("/");
	exercises->mkdir("ecalAndHcal");
	exercises->cd("/ecalAndHcal");
	evaluateSpaceManager(sm, elements_);
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		CalibratorPtr c = (*it).second;
		std::for_each(elements_.begin(), elements_.end(), resetElement);
		evaluateCalibrator(sm, c, tree, calibrated, ecal, hcal, offset, LINEAR,
				NONE);
		std::for_each(elements_.begin(), elements_.end(), resetElement);
	}
	sm->printCalibrations(std::cout);

	/* HCAL */
	std::cout << "*** Performace of HCAL ONLY calibration ***\n";
	getCalibrations(hsm);
	exercises->cd("/");
	exercises->mkdir("hcal");
	exercises->cd("/hcal");
	evaluateSpaceManager(hsm, elements_);
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = hsmCalibrators->begin(); it != hsmCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		CalibratorPtr c = (*it).second;
		std::for_each(elements_.begin(), elements_.end(), resetElement);
		evaluateCalibrator(hsm, c, tree, calibrated, ecal, hcal, offset,
				LINEAR, NONE);
		std::for_each(elements_.begin(), elements_.end(), resetElement);
	}
	hsm->printCalibrations(std::cout);

	/* ECAL */
	exercises->cd("/");
	std::cout << "*** Performace of ECAL ONLY calibration ***\n";
	getCalibrations(esm);
	exercises->cd("/");
	exercises->mkdir("ecal");
	exercises->cd("/ecal");
	evaluateSpaceManager(esm, elements_);
	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = esmCalibrators->begin(); it != esmCalibrators->end(); ++it) {
		SpaceVoxelPtr sv = (*it).first;
		CalibratorPtr c = (*it).second;
		std::for_each(elements_.begin(), elements_.end(), resetElement);
		evaluateCalibrator(esm, c, tree, calibrated, ecal, hcal, offset,
				LINEAR, NONE);
		std::for_each(elements_.begin(), elements_.end(), resetElement);
	}
	esm->printCalibrations(std::cout);

	exercises->cd("/");

	//Reevaluate correction parameters
	TF1* f1;
	TF1* f2;
	determineCorrection(*exercises, tree, f1, f2);

	//save results
	std::cout << "Writing output tree...\n";
	tree.Write();
	droppedParticles.Write();
	//gaussianFits(*exercises, calibVec);
	exercises->Write();
	exercises->Close();
	std::cout << "Done."<< std::endl;

}

void Exercises2::getCalibrations(SpaceManagerPtr s) {

	std::map<SpaceVoxelPtr, CalibratorPtr>* smCalibrators = s->getCalibrators();

	for (std::map<SpaceVoxelPtr, CalibratorPtr>::iterator
			it = smCalibrators->begin(); it != smCalibrators->end(); ++it) {
		CalibratorPtr c= (*it).second;
		std::for_each(elements_.begin(), elements_.end(), resetElement);
		if (c->hasParticles() > static_cast<int>(threshold_)) {
			std::map<DetectorElementPtr, double>
					calibs = c->getCalibrationCoefficients();
			s->assignCalibration(c, calibs);
		}
	}
}

void Exercises2::evaluateSpaceManager(SpaceManagerPtr s,
		std::vector<DetectorElementPtr> detEls) {

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
		if (s->getName() == "ecalOnly") {
			options_->GetOpt("evolution", "ecalOnlyEcalBarrel", ecalBarrel);
			options_->GetOpt("evolution", "ecalOnlyEcalEndcap", ecalEndcap);
			//assert(ecalBarrel.size() == 9 && ecalEndcap.size() == 9);
		} else if (s->getName() == "hcalOnly") {
			options_->GetOpt("evolution", "hcalOnlyHcalBarrel", hcalBarrel);
			options_->GetOpt("evolution", "hcalOnlyHcalEndcap", hcalEndcap);
			//assert(hcalBarrel.size() == 9 && hcalEndcap.size() == 9);
		} else {
			options_->GetOpt("evolution", "ecalHcalEcalBarrel", ecalBarrel);
			options_->GetOpt("evolution", "ecalHcalEcalEndcap", ecalEndcap);
			options_->GetOpt("evolution", "ecalHcalHcalBarrel", hcalBarrel);
			options_->GetOpt("evolution", "ecalHcalHcalEndcap", hcalEndcap);
			//assert(ecalBarrel.size() == 9 && ecalEndcap.size() == 9);
			//assert(hcalBarrel.size() == 9 && hcalEndcap.size() == 9);
		}

		for (std::vector<DetectorElementPtr>::iterator i = detEls.begin(); i
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
					//fBarrel(
						//	barrelName.c_str(),
							//"([0]*[5]*x*([1]-[5]*x)/pow(([2]+[5]*x),3)+[3]*pow([5]*x, 0.1))*([5]*x<[8] && [5]*x>[7])+[4]*([5]*x>[8])+([6]*[5]*x)*([5]*x<[7])");
			fBarrel(barrelName.c_str(),
										"([0]*[5]*x)*([5]*x<[1])+([2]+[3]*exp([4]*[5]*x))*([5]*x>[1])");


			if (d->getType() == ECAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator
						dit = ecalBarrel.begin(); dit!= ecalBarrel.end(); ++dit) {
					fBarrel.FixParameter(count, *dit);
					++count;
				}
				
			}
			if (d->getType() == HCAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator
						dit = hcalBarrel.begin(); dit!= hcalBarrel.end(); ++dit) {
					fBarrel.FixParameter(count, *dit);
					++count;
				}
				
			}
			if(useTruth)
				fBarrel.FixParameter(5, 1.0);

			fBarrel.SetMinimum(0);
			s->addEvolution(d, BARREL_POS, fBarrel);

			if (basePlots > 0) {
				TH1* slices = s->extractEvolution(d, BARREL_POS, fBarrel, useTruth);
				slices->Write();
			}
			fBarrel.Write();

			/* Fitting for endcap */
			std::string endcapName(name);
			endcapName.append(RegionNames[ENDCAP_POS]);
			std::cout << "\tFixing "<< RegionNames[ENDCAP_POS]<< "\n";
			TF1
				//	fEndcap(
					//		endcapName.c_str(),
						//	"([0]*[5]*x*([1]-[5]*x)/pow(([2]+[5]*x),3)+[3]*pow([5]*x, 0.1))*([5]*x<[8] && [5]*x>[7])+[4]*([5]*x>[8])+([6]*[5]*x)*([5]*x<[7])");
			fEndcap(endcapName.c_str(),
										"([0]*[5]*x)*([5]*x<[1])+([2]+[3]*exp([4]*[5]*x))*([5]*x>[1])");

			if (d->getType() == ECAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator
						dit = ecalEndcap.begin(); dit!= ecalEndcap.end(); ++dit) {
					fEndcap.FixParameter(count, *dit);
					++count;
				}
				
			}
			if (d->getType() == HCAL) {
				unsigned count(0);
				for (std::vector<double>::const_iterator
						dit = hcalEndcap.begin(); dit!= hcalEndcap.end(); ++dit) {
					fEndcap.FixParameter(count, *dit);
					++count;
				}
				
			}
			if(useTruth)
				fEndcap.FixParameter(5, 1.0);
			
			fEndcap.SetMinimum(0);
			s->addEvolution(d, ENDCAP_POS, fEndcap);
			if (basePlots > 0) {
				TH1* slices = s->extractEvolution(d, ENDCAP_POS, fEndcap, useTruth);
				slices->Write();
			}
			fEndcap.Write();
		}

	} else if (s->getNCalibrations() > 0) {
		std::cout << "Using autofit functionality...\n";
		for (std::vector<DetectorElementPtr>::iterator i = detEls.begin(); i
				!= detEls.end(); ++i) {
			DetectorElementPtr d = *i;

			std::string name("Func");

			name.append(DetElNames[d->getType()]);
			name.append("_");

			/* Fitting for barrel */
			std::string barrelName(name);
			barrelName.append(RegionNames[BARREL_POS]);
			std::cout << "\tFitting "<< RegionNames[BARREL_POS]<< "\n";
			TF1 fBarrel(barrelName.c_str(),
					"[0]*x*([1]-x)/pow(([2]+x),3)+[3]*pow(x, 0.1)");
			barrelName.append("Slices");

			TH1* slices = s->extractEvolution(d, BARREL_POS, fBarrel, useTruth);
			slices->Write();
			fBarrel.Write();

			if (slices != 0) {
				slices->SetName(barrelName.c_str());
				slices->Write();
				//				if (mode == 2) {
				//					//Use fit to truth rather than reco energies
				//					//(Clearly cheating and impossible with real data!)
				//					s->extractTruthEvolution(d, tg, &fBarrel);
				//				}
				s->addEvolution(d, BARREL_POS, fBarrel);
			} else {
				std::cout << __PRETTY_FUNCTION__
						<< ": WARNING! Couldn't get fitted slices!\n";
			}

			/* Fitting for endcaps */
			std::string endcapName(name.c_str());
			endcapName.append(RegionNames[ENDCAP_POS]);
			std::cout << "\nFitting "<< RegionNames[ENDCAP_POS]<< "\n";
			TF1 fEndcap(endcapName.c_str(),
					"[0]*x*([1]-x)/pow(([2]+x),3)+[3]*pow(x, 0.1)");

			endcapName.append("Slices");

			TH1* slicesEndcap = s->extractEvolution(d, ENDCAP_POS, fEndcap, useTruth);
			slicesEndcap->Write();
			fEndcap.Write();

			if (slicesEndcap != 0) {
				slicesEndcap->SetName(endcapName.c_str());
				slicesEndcap->Write();
				//				if (mode == 2) {
				//					//Use fit to truth rather than reco energies
				//					//(Clearly cheating and impossible with real data!)
				//					s->extractTruthEvolution(d, tg, &fEndcap);
				//				}
				s->addEvolution(d, ENDCAP_POS, fEndcap);
			} else {
				std::cout << __PRETTY_FUNCTION__
						<< ": WARNING! Couldn't get fitted slices!\n";
			}
		}
	}

}

void Exercises2::evaluateCalibrator(SpaceManagerPtr s, CalibratorPtr c,
		TTree& tree, Calibratable* calibrated, DetectorElementPtr ecal,
		DetectorElementPtr hcal, DetectorElementPtr offset,
		CalibrationProvenance cp, CalibrationProvenance cpCorr) {

	if (c->hasParticles() > static_cast<int>(threshold_)) {
		std::map<DetectorElementPtr, double>calibs = s->getCalibration(c);

		std::vector<ParticleDepositPtr> csParticles = c->getParticles();
		unsigned count(0);
		for (std::vector<ParticleDepositPtr >::iterator
				zit = csParticles.begin(); zit!= csParticles.end(); ++zit) {
			ParticleDepositPtr pd = *zit;
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
			crwPre.targetFuncContrib_ = pd->getTargetFunctionContrib();
			crwPre.target_ = target_;
			crwPre.compute();
			calibrated->calibrations_.push_back(crwPre);

			//evaluate calibration
			for (std::map<DetectorElementPtr, double>::iterator
					deit = calibs.begin(); deit != calibs.end(); ++deit) {
				DetectorElementPtr de = (*deit).first;

				int mode(0);
				options_->GetOpt("spaceManager", "interpolationMode", mode);

				if (mode == 1)
					de->setCalib(s->interpolateCoefficient(de,
							pd->getTruthEnergy(), pd->getEta(), pd->getPhi()));
				else if (mode == 2|| mode == 3|| mode == 4)
					de->setCalib(s->evolveCoefficient(de, pd->getRecEnergy(),
							pd->getEta(), pd->getPhi()));
				else
					de->setCalib((*deit).second);
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
						= clusterCalibration_.getCalibratedEcalEnergy(crwPre.ecalEnergy_,
								crwPre.hcalEnergy_, pd->getEta(), pd->getPhi());
				crwCorr.hcalEnergy_
						= clusterCalibration_.getCalibratedHcalEnergy(crwPre.ecalEnergy_,
								crwPre.hcalEnergy_, pd->getEta(), pd->getPhi());
				crwCorr.particleEnergy_
						= clusterCalibration_.getCalibratedEnergy(crwPre.ecalEnergy_,
								crwPre.hcalEnergy_, pd->getEta(), pd->getPhi());

				crwCorr.b_ = ecal->getCalib();
				crwCorr.c_ = hcal->getCalib();

				crwCorr.truthEnergy_ = pd->getTruthEnergy();
				crwCorr.provenance_ = cpCorr;
				crwCorr.targetFuncContrib_ = pd->getTargetFunctionContrib();
				crwCorr.target_ = target_;
				crwCorr.compute();
				calibrated->calibrations_.push_back(crwCorr);
			}
			tree.Fill();
			++count;

		}
	} 
}

void Exercises2::determineCorrection(TFile& f, TTree& tree, TF1*& f1, TF1*& f2) {
	std::cout << __PRETTY_FUNCTION__ << "\n";
	f.cd("/");
	f.mkdir("corrections");
	f.cd("/corrections");
	std::cout << "------------------------------------\nUncorrected curves:\n";
	tree.Draw(
			"sim_energyEvent_:calibrations_.particleEnergy_>>correctionCurve",
			"calibrations_.provenance_ > 0", "box");
	TH2D* correctionCurve = (TH2D*) gDirectory->Get("correctionCurve");
	correctionCurve->FitSlicesX();
	correctionCurve->Write();

	TH1F* correctionCurve_1 = (TH1F*) gDirectory->Get("correctionCurve_1");
	correctionCurve_1->Write();
	double correctionLowLimit(0);
	options_->GetOpt("exercises", "correctionLowLimit", correctionLowLimit);

	f1 = new TF1("f1", "pol1");
	correctionCurve_1->Fit("f1");
	f2 = new TF1("f2", "pol2");
	correctionCurve_1->Fit("f2");

	std::cout
			<< "------------------------------------\nAlready corrected curve fits:\n";
	tree.Draw(
			"sim_energyEvent_:calibrations_.particleEnergy_>>correctionCurveCorr",
			"calibrations_.provenance_ < 0", "box");
	TH2D* correctionCurveCorr = (TH2D*) gDirectory->Get("correctionCurveCorr");
	correctionCurveCorr->FitSlicesX();
	correctionCurveCorr->Write();

	TH1F
			* correctionCurveCorr_1 = (TH1F*) gDirectory->Get("correctionCurveCorr_1");
	correctionCurveCorr_1->Write();
	correctionCurveCorr_1->Fit("f1");
	correctionCurveCorr_1->Fit("f2");

	f.cd("/");
}
