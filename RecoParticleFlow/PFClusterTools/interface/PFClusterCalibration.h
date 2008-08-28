#ifndef PFCLUSTERCALIBRATION_H_
#define PFCLUSTERCALIBRATION_H_

#include "RecoParticleFlow/PFClusterTools/interface/IO.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElementType.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationResultWrapper.h"

#include <TF1.h>
#include <vector>
#include <string>
#include <map>
#include <TTree.h>

namespace pftools{

class PFClusterCalibration {
public:
	
	PFClusterCalibration(IO* options);
	
	virtual ~PFClusterCalibration() {};
	
	double getCalibratedEcalEnergy(double totalE, double ecalE, double hcalE, double eta, double phi);
	
	double getCalibratedHcalEnergy(double totalE, double ecalE, double hcalE, double eta, double phi);
	
	double getCalibratedEnergy(double totalE, double ecalE, double hcalE, double eta, double phi);
	
	const void calibrate(Calibratable& c);
	
	const void getCalibrationResultWrapper(const Calibratable& c, CalibrationResultWrapper& crw);
	
	void calibrateTree(TTree* tree);
	
private:
	
	IO* options_;
	//where to select either barrel or endcap
	double barrelEndcapEtaDiv_;
	
	//at what energy to split between ecalOnly, hcalOnly, ecalHcal
	double ecalOnlyDiv_;
	double hcalOnlyDiv_;
	
	//the energy below which we don't bother (a temporary solution to the <2GeV madness)
	double giveUpCut_;
	
	//what energy to consider calibration to be constant
	double flatlineEvoEnergy_;
	
	double correctionLowLimit_;
	double globalP0_;
	double globalP1_;
	double lowEP0_;
	double lowEP1_;
	
	//Function used to correct final total energies
	TF1 correction_;
	
	//function parameters
//	std::vector<double> ecalOnlyEcalBarrel_;
//	std::vector<double> ecalOnlyEcalEndcap_;
//	std::vector<double> hcalOnlyHcalBarrel_;
//	std::vector<double> hcalOnlyHcalEndcap_;
//	std::vector<double> ecalHcalEcalBarrel_;
//	std::vector<double> ecalHcalEcalEndcap_;
//	std::vector<double> ecalHcalHcalBarrel_;
//	std::vector<double> ecalHcalHcalEndcap_;
	
	std::map<std::string, TF1> namesAndFunctions_;
	std::vector<std::string> names_;
	
	//evolution functions
//	TF1 ecalOnlyEcalBarrel_;
//	TF1 FuncEcalEndcapEcalOnly_;
//	TF1 FuncHcalBarrelHcalOnly_;
//	TF1 FuncHcalEndcapHcalOnly_;
//	TF1 FuncEcalBarrelEcalHcal_;
//	TF1 FuncEcalEndcapEcalHcal_;
//	TF1 FuncHcalBarrelEcalHcal_;
//	TF1 FuncHcalEndcapEcalHcal_;
	
	
	
};
}

#endif /* PFCLUSTERCALIBRATION_H_ */
