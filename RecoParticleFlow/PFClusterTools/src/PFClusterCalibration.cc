#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationProvenance.h"

#include <cmath>
#include <cassert>
#include <TBranch.h>

using namespace pftools;

PFClusterCalibration::PFClusterCalibration(IO* options) :
	options_(options), correction_("correction",
			"((x-[0])/[1])*(x>[4])+((x-[2])/[3])*(x<[4])") {

	//read in and initialise!
	options_->GetOpt("evolution", "ecalECut", ecalOnlyDiv_);
	options_->GetOpt("evolution", "hcalECut", hcalOnlyDiv_);
	options_->GetOpt("evolution", "giveUpCut", giveUpCut_);
	options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEndcapEtaDiv_);
	options_->GetOpt("evolution", "evolutionFunctionMaxE", flatlineEvoEnergy_);

	options_->GetOpt("correction", "globalP0", globalP0_);
	options_->GetOpt("correction", "globalP1", globalP1_);
	options_->GetOpt("correction", "lowEP0", lowEP0_);
	options_->GetOpt("correction", "lowEP1", lowEP1_);
	options_->GetOpt("correction", "correctionLowLimit", correctionLowLimit_);

	correction_.FixParameter(0, globalP0_);
	correction_.FixParameter(1, globalP1_);
	correction_.FixParameter(2, lowEP0_);
	correction_.FixParameter(3, lowEP1_);
	correction_.FixParameter(4, correctionLowLimit_);

	std::string eoeb("ecalOnlyEcalBarrel");
	names_.push_back(eoeb);
	std::string eoee("ecalOnlyEcalEndcap");
	names_.push_back(eoee);
	std::string hohb("hcalOnlyHcalBarrel");
	names_.push_back(hohb);
	std::string hohe("hcalOnlyHcalEndcap");
	names_.push_back(hohe);

	std::string eheb("ecalHcalEcalBarrel");
	names_.push_back(eheb);
	std::string ehee("ecalHcalEcalEndcap");
	names_.push_back(ehee);
	std::string ehhb("ecalHcalHcalBarrel");
	names_.push_back(ehhb);
	std::string ehhe("ecalHcalHcalEndcap");
	names_.push_back(ehhe);

	char
			* funcString("([0]*[5]*x*([1]-[5]*x)/pow(([2]+[5]*x),3)+[3]*pow([5]*x, 0.1))*([5]*x<[6])+[4]*([5]*x>[6])");

	//Create functions for each sector
	for (std::vector<std::string>::const_iterator cit = names_.begin(); cit
			!= names_.end(); ++cit) {
		std::string name = *cit;
		TF1 func(name.c_str(), funcString);

		//Extract parameters from option file
		std::vector<double> params;
		options_->GetOpt("evolution", name.c_str(), params);
		unsigned count(0);
		std::cout << "Fixing for "<< name << "\n";
		for (std::vector<double>::const_iterator dit = params.begin(); dit
				!= params.end(); ++dit) {
			func.FixParameter(count, *dit);
			std::cout << "\t"<< count << ": "<< *dit << "\n";
			++count;
		}
		assert(count == 6);
		//Last parameters is common to all functions (for now).
		func.FixParameter(count, flatlineEvoEnergy_);
		//Store in map
		namesAndFunctions_[name] = func;

	}
}

double PFClusterCalibration::getCalibratedEcalEnergy(double totalE,
		double ecalE, double hcalE, double eta, double phi) {
	TF1* theFunction(0);
	if (ecalE > ecalOnlyDiv_ && hcalE > hcalOnlyDiv_) {
		//ecalHcal class
		if (fabs(eta) < barrelEndcapEtaDiv_) {
			//barrel
			theFunction = &namesAndFunctions_["ecalHcalEcalBarrel"];
		} else {
			//endcap
			theFunction = &namesAndFunctions_["ecalHcalEcalEndcap"];
		}
	} else if (ecalE > ecalOnlyDiv_ && hcalE < hcalOnlyDiv_) {
		//ecalOnly class
		if (fabs(eta) < barrelEndcapEtaDiv_)
			theFunction = &namesAndFunctions_["ecalOnlyEcalBarrel"];
		else
			theFunction = &namesAndFunctions_["ecalOnlyEcalEndcap"];
	} else {
		//either hcal only or too litte energy, in any case,
		return ecalE;
	}
	assert(theFunction != 0);
	double bCoeff = theFunction->Eval(totalE);
	return ecalE * bCoeff;
}

double PFClusterCalibration::getCalibratedHcalEnergy(double totalE,
		double ecalE, double hcalE, double eta, double phi) {
	TF1* theFunction(0);
	if (ecalE > ecalOnlyDiv_ && hcalE > hcalOnlyDiv_) {
		//ecalHcal class
		if (fabs(eta) < barrelEndcapEtaDiv_) {
			//barrel
			theFunction = &namesAndFunctions_["ecalHcalHcalBarrel"];
		} else {
			//endcap
			theFunction = &namesAndFunctions_["ecalHcalHcalEndcap"];
		}
	} else if (ecalE < ecalOnlyDiv_ && hcalE > hcalOnlyDiv_) {
		//hcalOnly class
		if (fabs(eta) < barrelEndcapEtaDiv_)
			theFunction = &namesAndFunctions_["hcalOnlyHcalBarrel"];
		else
			theFunction = &namesAndFunctions_["hcalOnlyHcalEndcap"];
	} else {
		//either ecal only or too litte energy, in any case,
		return hcalE;
	}
	assert(theFunction != 0);
	double cCoeff = theFunction->Eval(totalE);
	return hcalE * cCoeff;
}

double PFClusterCalibration::getCalibratedEnergy(double totalE, double ecalE,
		double hcalE, double eta, double phi) {
	double answer(totalE);
	if (totalE < giveUpCut_)
		return totalE;

	if (ecalE > ecalOnlyDiv_ && hcalE > hcalOnlyDiv_) {
		//ecalHcal class
		answer = getCalibratedEcalEnergy(totalE, ecalE, hcalE, eta, phi)
				+ getCalibratedHcalEnergy(totalE, ecalE, hcalE, eta, phi);
	} else if (ecalE < ecalOnlyDiv_ && hcalE > hcalOnlyDiv_) {
		//hcalOnly class
		answer = getCalibratedHcalEnergy(totalE, ecalE, hcalE, eta, phi);
	} else if (ecalE > ecalOnlyDiv_ && hcalE < hcalOnlyDiv_) {
		//ecalOnly
		answer = getCalibratedEcalEnergy(totalE, ecalE, hcalE, eta, phi);
	} else {
		//else, too little energy, give up
		return answer;
	}

	//apply correction
	return correction_.Eval(answer);
}

const void PFClusterCalibration::calibrate(Calibratable& c) {
	CalibrationResultWrapper crw;
	getCalibrationResultWrapper(c, crw);
	c.calibrations_.push_back(crw);

}

const void PFClusterCalibration::getCalibrationResultWrapper(
		const Calibratable& c, CalibrationResultWrapper& crw) {

	double totalE = c.cluster_energyEcal_ + c.cluster_energyHcal_;

	crw.ecalEnergy_ = getCalibratedEcalEnergy(totalE, c.cluster_energyEcal_,
			c.cluster_energyHcal_, fabs(c.cluster_meanEcal_.eta_),
			fabs(c.cluster_meanEcal_.phi_));

	crw.hcalEnergy_ = getCalibratedHcalEnergy(totalE, c.cluster_energyEcal_,
			c.cluster_energyHcal_, fabs(c.cluster_meanEcal_.eta_),
			fabs(c.cluster_meanEcal_.phi_));

	crw.particleEnergy_ = getCalibratedEnergy(totalE, c.cluster_energyEcal_,
			c.cluster_energyHcal_, fabs(c.cluster_meanEcal_.eta_),
			fabs(c.cluster_meanEcal_.phi_));

	crw.provenance_ = LINEARCORR;
	crw.b_ = crw.ecalEnergy_ / c.cluster_energyEcal_;
	crw.c_ = crw.hcalEnergy_ / c.cluster_energyHcal_;
	crw.truthEnergy_ = c.sim_energyEvent_;

}

void PFClusterCalibration::calibrateTree(TTree* input) {
	std::cout << __PRETTY_FUNCTION__ << ": WARNING! This isn't working properly yet!\n";
	TBranch* calibBr = input->GetBranch("Calibratable");
	Calibratable* calib_ptr = new Calibratable();
	calibBr->SetAddress(&calib_ptr);
	
	TBranch* newBranch = input->Branch("NewCalibratable", "pftools::Calibratable", &calib_ptr, 32000, 2);
	
	std::cout << "Looping over tree's "<< input->GetEntries() << " entries...\n";
	for (unsigned entries(0); entries < 20000; entries++) {
		if(entries % 10000 == 0)
			std::cout << "\tProcessing entry " << entries << "\n";
		input->GetEntry(entries);
		calibrate(*calib_ptr);
		input->Fill();
	}
	//input.Write("",TObject::kOverwrite);
}

