
{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");

	
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	std::cout << "Constructing IO..." << std::endl;

	IO* options_ = new IO("pfClusterTools.opt");
	
	PFClusterCalibration clusterCalibration_;
	
	//Initialise function parameters properly.
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
	
	std::vector<std::string>* names = clusterCalibration_.getKnownSectorNames();
	for(std::vector<std::string>::iterator i = names->begin(); i != names->end(); ++i) {
		std::string sector = *i;
		std::vector<double> params;
		options_->GetOpt("evolution", sector.c_str(), params);
		clusterCalibration_.setEvolutionParameters(sector, params);
	}
	
	std::cout << clusterCalibration_ << "\n";
	
}
