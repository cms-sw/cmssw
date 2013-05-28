#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"

#include <cmath>
#include <cassert>
#include <TBranch.h>
#include <TF1.h>
#include <TTree.h>

using namespace pftools;

void PFClusterCalibration::init() {
  
  //std::cout << __PRETTY_FUNCTION__ << std::endl;
	correction_ = new TF1("correction",
			"((x-[0])/[1])*(x>[4])+((x-[2])/[3])*(x<[4])");
	etaCorrection_
			= new TF1( "etaCorrection",
					"(1-[0]*x-[1]*x*x)*(x<[2])+([3]+[4]*x)*(x>[2]&&x<[5])+(1-[6]*x-[7]*x*x)*(x>[5])");

	correction_->FixParameter(0, globalP0_);
	correction_->FixParameter(1, globalP1_);
	correction_->FixParameter(2, lowEP0_);
	correction_->FixParameter(3, lowEP1_);
	correction_->FixParameter(4, correctionLowLimit_);

	/* These are the types of calibration I know about:
	 * ecalOnly_elementName
	 * etc. Sorry, it's not very nice, but well, neither is ROOT... */

	std::string eheb("ecalHcalEcalBarrel");
	names_.push_back(eheb);
	std::string ehee("ecalHcalEcalEndcap");
	names_.push_back(ehee);
	std::string ehhb("ecalHcalHcalBarrel");
	names_.push_back(ehhb);
	std::string ehhe("ecalHcalHcalEndcap");
	names_.push_back(ehhe);

	/* char
	 * funcString("([0]*[5]*x*([1]-[5]*x)/pow(([2]+[5]*x),3)+[3]*pow([5]*x, 0.1))*([5]*x<[8] && [5]*x>[7])+[4]*([5]*x>[8])+([6]*[5]*x)*([5]*x<[7])");
	 */

	const char*
	  funcString("([0]*[5]*x)*([5]*x<=[1])+([2]+[3]*exp([4]*[5]*x))*([5]*x>[1])");

	//Create functions for each sector
	for (std::vector<std::string>::const_iterator cit = names_.begin(); cit
			!= names_.end(); ++cit) {
		std::string name = *cit;
		TF1 func(name.c_str(), funcString);
		//some sensible defaults
		func.FixParameter(0, 1);
		func.FixParameter(1, 0);
		func.FixParameter(2, 1);
		func.FixParameter(3, 0);
		func.FixParameter(4, 0);
		func.FixParameter(5, 1);

		func.SetMinimum(0);
		//Store in map
		namesAndFunctions_[name] = func;

	}
}

/*PFClusterCalibration::PFClusterCalibration(IO* options_) :
	barrelEndcapEtaDiv_(1.0), ecalOnlyDiv_(0.3), hcalOnlyDiv_(0.5),
			doCorrection_(1), allowNegativeEnergy_(0), doEtaCorrection_(1),
			maxEToCorrect_(-1.0), correctionLowLimit_(0.), globalP0_(0.0),
			globalP1_(1.0), lowEP0_(0.0), lowEP1_(1.0) {

	init();

	double g0, g1, e0, e1;
	options_->GetOpt("correction", "globalP0", g0);
	options_->GetOpt("correction", "globalP1", g1);
	options_->GetOpt("correction", "lowEP0", e0);
	options_->GetOpt("correction", "lowEP1", e1);
	setCorrections(e0, e1, g0, g1);

	options_->GetOpt("correction", "allowNegativeEnergy", allowNegativeEnergy_);
	options_->GetOpt("correction", "doCorrection", doCorrection_);
	options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEndcapEtaDiv_);

	std::vector<std::string>* names = getKnownSectorNames();
	for (std::vector<std::string>::iterator i = names->begin(); i
			!= names->end(); ++i) {
		std::string sector = *i;
		std::vector<double> params;
		options_->GetOpt("evolution", sector.c_str(), params);
		setEvolutionParameters(sector, params);
	}

	options_->GetOpt("evolution", "doEtaCorrection", doEtaCorrection_);

	std::vector<double> etaParams;
	options_->GetOpt("evolution", "etaCorrection", etaParams);
	setEtaCorrectionParameters(etaParams);

} */

PFClusterCalibration::PFClusterCalibration() :
	barrelEndcapEtaDiv_(1.0), ecalOnlyDiv_(0.3), hcalOnlyDiv_(0.5),
			doCorrection_(1), allowNegativeEnergy_(0), doEtaCorrection_(1),
			maxEToCorrect_(-1.0), correctionLowLimit_(0.), globalP0_(0.0),
			globalP1_(1.0), lowEP0_(0.0), lowEP1_(1.0) {
  //	std::cout << __PRETTY_FUNCTION__ << std::endl;
	init();
	//	std::cout
	//			<< "WARNING! PFClusterCalibration evolution functions have not yet been initialised - ensure this is done.\n";
	//	std::cout << "PFClusterCalibration construction complete."<< std::endl;

}

PFClusterCalibration::~PFClusterCalibration() {
	delete correction_;
	delete etaCorrection_;
}

void PFClusterCalibration::setEtaCorrectionParameters(const std::vector<double>& params) {
	if (params.size() != 6) {
		std::cout << __PRETTY_FUNCTION__ << ": params is of the wrong length."
				<< std::endl;
		return;
	}
	//	std::cout << "Fixing eta correction:\n\t";
	unsigned count(0);
	for (std::vector<double>::const_iterator dit = params.begin(); dit
			!= params.end(); ++dit) {
	  //std::cout << *dit << "\t";
		etaCorrection_->FixParameter(count, *dit);
		++count;
	}
	//	std::cout << std::endl;
	/*for(double eta(0); eta < 2.5; eta += 0.05) {
	 std::cout << "Eta = " << eta << ",\tcorr = " << etaCorrection_->Eval(eta) << "\n"; 
	 }*/
}

void PFClusterCalibration::setEvolutionParameters(const std::string& sector,
		const std::vector<double>& params) {
	TF1* func = &(namesAndFunctions_.find(sector)->second);
	unsigned count(0);
	//std::cout << "Fixing for "<< sector << "\n";
	for (std::vector<double>::const_iterator dit = params.begin(); dit
			!= params.end(); ++dit) {
		func->FixParameter(count, *dit);
		//std::cout << "\t"<< count << ": "<< *dit;
		++count;
	}
	//	std::cout << std::endl;
	func->SetMinimum(0);
}

void PFClusterCalibration::setCorrections(const double& lowEP0,
		const double& lowEP1, const double& globalP0, const double& globalP1) {
	//'a' term is -globalP0/globalP1
	globalP0_ = globalP0;
	globalP1_ = globalP1;
	//Specifically for low energies...
	lowEP0_ = lowEP0;
	lowEP1_ = lowEP1;
	//Intersection of two straight lines => matching at...
	correctionLowLimit_ = (lowEP0_ - globalP0_)/(globalP1_ - lowEP1_);

	correction_->FixParameter(0, globalP0_);
	correction_->FixParameter(1, globalP1_);
	correction_->FixParameter(2, lowEP0_);
	correction_->FixParameter(3, lowEP1_);
	correction_->FixParameter(4, correctionLowLimit_);

	//	std::cout << __PRETTY_FUNCTION__ << ": setting correctionLowLimit_ = "
	//		<< correctionLowLimit_ << "\n";
}

double PFClusterCalibration::getCalibratedEcalEnergy(const double& ecalE,
		const double& hcalE, const double& eta, const double& phi) const {
	const TF1* theFunction(0);

	if (fabs(eta) < barrelEndcapEtaDiv_) {
		//barrel
		theFunction = &(namesAndFunctions_.find("ecalHcalEcalBarrel")->second);
	} else {
		//endcap
		theFunction = &(namesAndFunctions_.find("ecalHcalEcalEndcap")->second);
	}

	assert(theFunction != 0);
	double totalE(ecalE + hcalE);
	double bCoeff = theFunction->Eval(totalE);
	return ecalE * bCoeff;
}

double PFClusterCalibration::getCalibratedHcalEnergy(const double& ecalE,
		const double& hcalE, const double& eta, const double& phi) const {
	const TF1* theFunction(0);

	if (fabs(eta) < barrelEndcapEtaDiv_) {
		//barrel
		theFunction = &(namesAndFunctions_.find("ecalHcalHcalBarrel")->second);
	} else {
		//endcap
		theFunction = &(namesAndFunctions_.find("ecalHcalHcalEndcap")->second);
	}

	double totalE(ecalE + hcalE);
	assert(theFunction != 0);
	double cCoeff = theFunction->Eval(totalE);
	return hcalE * cCoeff;
}

double PFClusterCalibration::getCalibratedEnergy(const double& ecalE,
		const double& hcalE, const double& eta, const double& phi) const {
	double totalE(ecalE + hcalE);
	double answer(totalE);

	answer = getCalibratedEcalEnergy(ecalE, hcalE, eta, phi)
			+ getCalibratedHcalEnergy(ecalE, hcalE, eta, phi);
	if (doEtaCorrection_)
		answer = answer/etaCorrection_->Eval(eta);

	if (maxEToCorrect_> 0 && answer < maxEToCorrect_)
		return correction_->Eval(answer);
	if (doCorrection_) {
		if (maxEToCorrect_> 0 && answer < maxEToCorrect_)
			answer = correction_->Eval(answer);
		else if (maxEToCorrect_ < 0) {
			answer = correction_->Eval(answer);
		}
	}
	if (!allowNegativeEnergy_ && answer < 0)
		return 0;
	return answer;

}

void PFClusterCalibration::getCalibratedEnergyEmbedAInHcal(double& ecalE,
		double& hcalE, const double& eta, const double& phi) const {

	double ecalEOld(ecalE);
	double hcalEOld(hcalE);

	ecalE = getCalibratedEcalEnergy(ecalEOld, hcalEOld, eta, phi);
	hcalE = getCalibratedHcalEnergy(ecalEOld, hcalEOld, eta, phi);

	double preCorrection(ecalE + hcalE);
	if (doEtaCorrection_)
		preCorrection = preCorrection/etaCorrection_->Eval(eta);

	if (doCorrection_) {
		double corrE = correction_->Eval(preCorrection);
		//a term  = difference
		double a = corrE - preCorrection;
		hcalE += a;
	}
	if (hcalE < 0 && !allowNegativeEnergy_)
		hcalE = 0;

}

void PFClusterCalibration::calibrate(Calibratable& c) {
	CalibrationResultWrapper crw;
	getCalibrationResultWrapper(c, crw);
	c.calibrations_.push_back(crw);

}

void PFClusterCalibration::getCalibrationResultWrapper(const Calibratable& c,
		CalibrationResultWrapper& crw) {

	crw.ecalEnergy_ = getCalibratedEcalEnergy(c.cluster_energyEcal_,
			c.cluster_energyHcal_, fabs(c.cluster_meanEcal_.eta_),
			fabs(c.cluster_meanEcal_.phi_));

	crw.hcalEnergy_ = getCalibratedHcalEnergy(c.cluster_energyEcal_,
			c.cluster_energyHcal_, fabs(c.cluster_meanEcal_.eta_),
			fabs(c.cluster_meanEcal_.phi_));

	crw.particleEnergy_ = getCalibratedEnergy(c.cluster_energyEcal_,
			c.cluster_energyHcal_, fabs(c.cluster_meanEcal_.eta_),
			fabs(c.cluster_meanEcal_.phi_));

	crw.provenance_ = LINEARCORR;
	crw.b_ = crw.ecalEnergy_ / c.cluster_energyEcal_;
	crw.c_ = crw.hcalEnergy_ / c.cluster_energyHcal_;
	crw.truthEnergy_ = c.sim_energyEvent_;

}

void PFClusterCalibration::calibrateTree(TTree* input) {
	std::cout << __PRETTY_FUNCTION__
			<< ": WARNING! This isn't working properly yet!\n";
	TBranch* calibBr = input->GetBranch("Calibratable");
	Calibratable* calib_ptr = new Calibratable();
	calibBr->SetAddress(&calib_ptr);

	//	TBranch* newBranch = input->Branch("NewCalibratable",
	//			"pftools::Calibratable", &calib_ptr, 32000, 2);

	std::cout << "Looping over tree's "<< input->GetEntries()
			<< " entries...\n";
	for (unsigned entries(0); entries < 20000; entries++) {
		if (entries % 10000== 0)
			std::cout << "\tProcessing entry "<< entries << "\n";
		input->GetEntry(entries);
		calibrate(*calib_ptr);
		input->Fill();
	}
	//input.Write("",TObject::kOverwrite);
}

std::ostream& pftools::operator<<(std::ostream& s, const PFClusterCalibration& cc) {
	s << "PFClusterCalibration: dump...\n";
	s << "barrelEndcapEtaDiv:\t" << cc.barrelEndcapEtaDiv_ << ", ecalOnlyDiv:\t" << cc.ecalOnlyDiv_;
	s << ", \nhcalOnlyDiv:\t" << cc.hcalOnlyDiv_ << ", doCorrection:\t" << cc.doCorrection_;
	s << ", \nallowNegativeEnergy:\t" << cc.allowNegativeEnergy_;
	s << ", \ncorrectionLowLimit:\t" << cc.correctionLowLimit_ << ",parameters:\t" << cc.globalP0_ << ", ";
	s << cc.globalP1_ << ", " << cc.lowEP0_ << ", " << cc.lowEP1_;
	s << "\ndoEtaCorrection:\t" << cc.doEtaCorrection_;
	return s;
}

