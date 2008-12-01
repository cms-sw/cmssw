#ifndef EXERCISES2_H_
#define EXERCISES2_H_

#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "RecoParticleFlow/PFClusterTools/interface/IO.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"

#include <string>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <fstream>

namespace pftools {
class Exercises2 {
public:
		
	Exercises2(IO* options);
	
	virtual ~Exercises2();

	void calibrateCalibratables(const std::string& sourcefile,
			const std::string& exercisefile);

	//void gaussianFits(TFile& exercisefile, std::vector<Calibratable>& calibs);

	void evaluateCalibrator(SpaceManagerPtr s, CalibratorPtr c, TTree& tree,
			Calibratable* calibrated, DetectorElementPtr ecal,
			DetectorElementPtr hcal, DetectorElementPtr offset, CalibrationProvenance cp, CalibrationProvenance cpCorr = NONE);

	
	void evaluateSpaceManager(SpaceManagerPtr s, std::vector<DetectorElementPtr> detEls);

	void determineCorrection(TFile& f, TTree& tree, TF1*& f1, TF1*& f2);
	
	void setTarget(CalibrationTarget t) {
		target_ = t;
	}
	
	
	void getCalibrations(SpaceManagerPtr s);

private:
	
	Exercises2(const Exercises2&);
	void operator=(const Exercises2&);
//	double lowE_, highE_, lowEta_, highEta_, lowPhi_, highPhi_;
//	unsigned divE_, divEta_, divPhi_;
	bool withOffset_;
	CalibrationTarget target_;
	unsigned threshold_;
	std::vector<DetectorElementPtr> elements_;
	IO* options_;
	std::ofstream calibResultsFile_;
	unsigned debug_;
	PFClusterCalibration clusterCalibration_;

};
}

#endif /*EXERCISES2_H_*/
