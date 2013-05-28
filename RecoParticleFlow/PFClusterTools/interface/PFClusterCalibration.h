#ifndef PFCLUSTERCALIBRATION_H_
#define PFCLUSTERCALIBRATION_H_

#include "RecoParticleFlow/PFClusterTools/interface/DetectorElementType.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationResultWrapper.h"
//#include "RecoParticleFlow/PFClusterTools/interface/IO.h"

#include <vector>
#include <string>
#include <map>
#include <ostream>
#include <iosfwd>

class TF1;
class TTree;

namespace pftools {

class PFClusterCalibration;
std::ostream& operator<<(std::ostream& s, const PFClusterCalibration& cc);

/*
 * \class PFClusterCalibration
 * \brief Calibrated calorimeter cluster energy for hadronic PFCandidates.
 * \author Jamie Ballin, Imperial College London
 * \date September 2008
 * 
 * The ECAL and HCAL have been calibrated to 50 GeV electrons. Therefore, a calibration is required 
 * to evaluate the correct hadronic response. This class will calibrate clusters belonging to a PFCandidate.
 * (Users should access these clusters from the blocks in the PFCandidate).
 * 
 * A linear calibration is evaluated, for barrel and endcap (call setBarrelBoundary(double eta)
 * to set this limit).
 * 
 * Sensible default values are set for all members, but in order to get usable results, you must supply the
 * latest function parameters and corrections (seperately available) - see setCorrections() 
 * and setEvolutionParameters() documentation below.
 */
class PFClusterCalibration {
public:
	
	/* Constructor with sensible defaults */
	PFClusterCalibration();
	
	//PFClusterCalibration(IO* io);

	virtual ~PFClusterCalibration();

	/* Returns the calibrated ecalEnergy */
	double getCalibratedEcalEnergy(const double& ecalE, const double& hcalE,
			const double& eta, const double& phi) const;
	
	/* Returns the calibrated hcalEnergy */
	double getCalibratedHcalEnergy(const double& ecalE, const double& hcalE,
			const double& eta, const double& phi) const;

	/* DEPRECATED METHOD - do not use.
	 * 
	 * Returns the calibrated particle energy with the correction
	 * Note: for, say, ecalOnly particles:
	 * energy = correction_function([calibrated ecalEnergy + hcalEnergy(v small)])
	 * ditto hcalOnly
	 */
	double getCalibratedEnergy(const double& ecalE, const double& hcalE,
			const double& eta, const double& phi) const;
	
	void getCalibratedEnergyEmbedAInHcal(double& ecalE,
			double& hcalE, const double& eta, const double& phi) const;

	/* TESTING purposes only! */
	void calibrate(Calibratable& c);

	/* TESTING purposes only! */
	void getCalibrationResultWrapper(const Calibratable& c,
			CalibrationResultWrapper& crw);

	/* TESTING purposes only! */
	void calibrateTree(TTree* tree);

	/* Sets the 'a' term in the abc calibration and a final linear correction.
	 * You get these values from the (seperately available) option file. */
	void setCorrections(const double& lowEP0, const double& lowEP1,
			const double& globalP0, const double& globalP1);

	/* getCalibratedEnergy() returns max(0, calibrated energy) if this is true. */
	void setAllowNegativeEnergy(const bool& allowIt) {
		allowNegativeEnergy_ = allowIt;
	}

	/* Whether to apply a final correction function in getCalibratedEnergy()
	 * Highly recommended ('a' term of abc calibration will be neglected otherwise. */
	void setDoCorrection(const int& doCorrection) {
		doCorrection_ = doCorrection;
	}
	
	void setDoEtaCorrection(const int doEtaCorrection) {
		doEtaCorrection_ = doEtaCorrection;
	}

	/* Threshold for ecalOnly and hcalOnly evaluation. */
	void setEcalHcalEnergyCuts(const double& ecalCut, const double& hcalCut) {
		//std::cout << __PRETTY_FUNCTION__ << "WARNING! These will be ignored.\n";
		ecalOnlyDiv_ = ecalCut;
		hcalOnlyDiv_ = hcalCut;
	}

	/* Hard cut between barrel and endcap. */
	void setBarrelBoundary(const double& eta) {
		barrelEndcapEtaDiv_ = eta;
	}
	
	void setMaxEToCorrect(double maxE) {
		maxEToCorrect_ = maxE;
	}

	/* Sets the function parameters - very important! */
	void setEvolutionParameters(const std::string& sector,
			const std::vector<double>& params);
	
	void setEtaCorrectionParameters(const std::vector<double>& params);

	/* Elements in this vector refer to the different calibration functions
	 * available. For each one of these, you should call setEvolutionParameters()
	 * with the appropriate vector<double> acquired from an options file.
	 */
	std::vector<std::string>* getKnownSectorNames() {
		return &names_;
	}

	
	/* Dumps the member values to the stream */
	friend std::ostream& pftools::operator<<(std::ostream& s, const PFClusterCalibration& cc);

private:
	
	void init();

	//where to select either barrel or endcap
	double barrelEndcapEtaDiv_;

	//at what energy to split between ecalOnly, hcalOnly, ecalHcal
	double ecalOnlyDiv_;
	double hcalOnlyDiv_;

	int doCorrection_;
	int allowNegativeEnergy_;
	int doEtaCorrection_;
	double maxEToCorrect_;

	double correctionLowLimit_;
	double globalP0_;
	double globalP1_;
	double lowEP0_;
	double lowEP1_;

	//Function used to correct final total energies
	TF1* correction_;
	//Function to correct eta dependence (post-calibration).
	TF1* etaCorrection_;

	std::map<std::string, TF1> namesAndFunctions_;
	std::vector<std::string> names_;

};
}

#endif /* PFCLUSTERCALIBRATION_H_ */
