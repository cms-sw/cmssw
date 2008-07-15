#ifndef EXERCISES2_H_
#define EXERCISES2_H_

#include "RecoParticleFlow/PFClusterTools/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationProvenance.h"
#include <string>
#include <vector>
#include <TFile.h>
#include <TTree.h>

namespace pftools {
class Exercises2 {
public:
	
	

	Exercises2(double lowE = 0, double highE = 100, unsigned divE = 1,
			double lowEta = -5, double highEta = 5, double divEta = 1,
			double lowPhi = -3.2, double highPhi = 3.2, unsigned divPhi = 1,
			bool withOffset = false);

	virtual ~Exercises2();

	void calibrateCalibratables(const std::string& sourcefile,
			const std::string& exercisefile);

	void gaussianFits(TFile& exercisefile, std::vector<Calibratable>& calibs);

	void evaluateCalibrator(CalibratorPtr c, TTree& tree,
			Calibratable* calibrated, DetectorElementPtr ecal,
			DetectorElementPtr hcal, CalibrationProvenance cp);

	//makes performance comparisons for the CalibrationResultWrappers in the source file,
	//and compares calibrations for each target supplied
	void doPlots(const std::string& sourcefile,
			std::vector<CalibrationTarget>& targets);
	
	void setTarget(CalibrationTarget t) {
		target_ = t;
	}

private:
	double lowE_, highE_, lowEta_, highEta_, lowPhi_, highPhi_;
	unsigned divE_, divEta_, divPhi_;
	bool withOffset_;
	CalibrationTarget target_;

};
}

#endif /*EXERCISES2_H_*/
