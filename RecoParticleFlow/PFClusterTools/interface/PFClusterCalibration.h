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
	
	bool doCorrection_;
	
	//the energy below which we don't bother (a temporary solution to the <2GeV madness)
	double giveUpCut_;
	
	//what energy to consider calibration to be constant
	double flatlineEvoEnergy_;
	
	double correctionLowLimit_;
	double correctionSuperLowLimit_;
	double globalP0_;
	double globalP1_;
	double lowEP0_;
	double lowEP1_;
	double superLowEP1_;
	
	//Function used to correct final total energies
	TF1 correction_;
	
	std::map<std::string, TF1> namesAndFunctions_;
	std::vector<std::string> names_;

	
	
	
};
}

#endif /* PFCLUSTERCALIBRATION_H_ */
