#ifndef EXERCISES3_H_
#define EXERCISES3_H_

#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"

#include <string>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <fstream>


namespace pftools {
class IO;

class Exercises3 {
public:
		
	Exercises3(IO* options);
	
	virtual ~Exercises3();
	
	void calibrateCalibratables(TChain& sourceTree,
				const std::string& exercisefile);

	//void gaussianFits(TFile& exercisefile, std::vector<Calibratable>& calibs);

	void evaluateCalibrator(SpaceManagerPtr s, CalibratorPtr c, TTree& tree,
			Calibratable* calibrated, DetectorElementPtr ecal,
			DetectorElementPtr hcal, DetectorElementPtr offset, CalibrationProvenance cp, CalibrationProvenance cpCorr = NONE);

	
	void evaluateSpaceManager(SpaceManagerPtr s, const std::vector<DetectorElementPtr>& detEls);

	
	void setTarget(CalibrationTarget t) {
		target_ = t;
	}
	
	
	void getCalibrations(SpaceManagerPtr s);

private:
	
	Exercises3(const Exercises3&);
	void operator=(const Exercises3&);
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

#endif /*EXERCISES3_H_*/
