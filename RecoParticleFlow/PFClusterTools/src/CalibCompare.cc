#include "RecoParticleFlow/PFClusterTools/interface/CalibCompare.h"
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

CalibCompare::~CalibCompare() {
}

CalibCompare::CalibCompare(IO* options) :
	withOffset_(false), target_(CLUSTER), options_(options), debug_(0),
			mlpOffset_(0.0), mlpSlope_(1.0) {

	options_->GetOpt("exercises", "withOffset", withOffset_);
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

	options_->GetOpt("correction", "mlpOffset", mlpOffset_);
	options_->GetOpt("correction", "mlpSlope", mlpSlope_);

	erlCalibration_.setOffsetAndSlope(mlpOffset_, mlpSlope_);

	if (debug_ > 0)
		std::cout << __PRETTY_FUNCTION__ << ": finished.\n";

}

void CalibCompare::calibrateCalibratables(TChain& sourceTree,
		const std::string& exercisefile) {

	if (debug_ > 0) {
		std::cout << "Welcome to " << __PRETTY_FUNCTION__ << "\n";
		std::cout << "Opening TTree...\n";
	}

	TreeUtility tu;
	std::vector<Calibratable> calibVec;

	tu.getCalibratablesFromRootFile(sourceTree, calibVec);

	std::cout << "Moving on... " << std::endl;
	TFile* exercises = new TFile(exercisefile.c_str(), "recreate");
	TTree tree("CalibratedParticles", "");
	Calibratable* calibrated = new Calibratable();
	tree.Branch("Calibratable", "pftools::Calibratable", &calibrated, 32000, 2);

	evaluateCalibrations(tree, calibrated, calibVec);

	//save results
	std::cout << "Writing output tree...\n";
	tree.Write();
	//gaussianFits(*exercises, calibVec);
	exercises->Write();
	exercises->Close();
	std::cout << "Done." << std::endl;
	delete calibrated;

}

void CalibCompare::evaluateCalibrations(TTree& tree, Calibratable* calibrated,
		const std::vector<Calibratable>& calibVec) {

	unsigned count(0);
	for (std::vector<Calibratable>::const_iterator zit = calibVec.begin(); zit
			!= calibVec.end(); ++zit) {

		const Calibratable& calib = *zit;

		calibrated->reset();

		CalibrationResultWrapper crwPre;
		crwPre.ecalEnergy_ = calib.cluster_energyEcal_;
		crwPre.hcalEnergy_ = calib.cluster_energyHcal_;
		crwPre.particleEnergy_ = calib.cluster_energyEcal_
				+ calib.cluster_energyHcal_;
		crwPre.truthEnergy_ = calib.sim_energyEvent_;
		crwPre.provenance_ = UNCALIBRATED;
		crwPre.targetFuncContrib_ = 0;
		crwPre.target_ = target_;
		crwPre.compute();
		calibrated->calibrations_.push_back(crwPre);

		CalibrationResultWrapper crwErl;

		crwErl.particleEnergy_ = erlCalibration_.evaluate(crwPre.ecalEnergy_,
				crwPre.hcalEnergy_, calib.cluster_numEcal_,
				calib.cluster_numHcal_, fabs(calib.cluster_meanEcal_.eta_)
						/ 2.0, crwPre.ecalEnergy_ / (crwPre.particleEnergy_),
				(calib.cluster_meanEcal_.phi_ + 3.14) / 6.3);
		crwErl.ecalEnergy_ = crwErl.particleEnergy_
				* erlCalibration_.ecalFraction(crwPre.ecalEnergy_,
						crwPre.hcalEnergy_, calib.cluster_numEcal_,
						calib.cluster_numHcal_, fabs(
								calib.cluster_meanEcal_.eta_) / 2.0,
						crwPre.ecalEnergy_ / (crwPre.particleEnergy_),
						(calib.cluster_meanEcal_.phi_ + 3.14) / 6.3);

		crwErl.hcalEnergy_ = crwErl.particleEnergy_ - crwErl.ecalEnergy_;
		crwErl.b_ = crwErl.ecalEnergy_ / crwPre.ecalEnergy_;
		crwErl.c_ = crwErl.hcalEnergy_ / crwPre.hcalEnergy_;

		crwErl.truthEnergy_ = calib.sim_energyEvent_;
		crwErl.provenance_ = BAYESIAN;
		crwErl.target_ = target_;
		crwErl.compute();
		calibrated->calibrations_.push_back(crwErl);

		CalibrationResultWrapper crwCorr;

		crwCorr.ecalEnergy_ = clusterCalibration_.getCalibratedEcalEnergy(
				crwPre.ecalEnergy_, crwPre.hcalEnergy_,
				calib.cluster_meanEcal_.eta_, calib.cluster_meanEcal_.phi_);
		crwCorr.hcalEnergy_ = clusterCalibration_.getCalibratedHcalEnergy(
				crwPre.ecalEnergy_, crwPre.hcalEnergy_,
				calib.cluster_meanHcal_.eta_, calib.cluster_meanHcal_.phi_);
		crwCorr.particleEnergy_ = clusterCalibration_.getCalibratedEnergy(
				crwPre.ecalEnergy_, crwPre.hcalEnergy_, calib.sim_etaEcal_,
				calib.sim_phiEcal_);

		crwCorr.truthEnergy_ = calib.sim_energyEvent_;
		crwCorr.provenance_ = LINEAR;
		crwCorr.targetFuncContrib_ = 0;;
		crwCorr.target_ = target_;
		crwCorr.compute();
		calibrated->calibrations_.push_back(crwCorr);

		calibrated->recompute();

		tree.Fill();

		++count;
	}

}

