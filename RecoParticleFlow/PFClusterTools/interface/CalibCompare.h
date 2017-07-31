#ifndef CALIBCOMPARE_H_
#define CALIBCOMPARE_H_

#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/Erl_mlp.h"

#include <string>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <fstream>


namespace pftools {
class IO;

class CalibCompare {
public:

	CalibCompare(IO* options);

	virtual ~CalibCompare();

	void calibrateCalibratables(TChain& sourceTree,
				const std::string& exercisefile);


	void setTarget(CalibrationTarget t) {
		target_ = t;
	}


	void evaluateCalibrations(TTree& tree, pftools::Calibratable* calibrated, const std::vector<pftools::Calibratable>& calibVec);

private:

	CalibCompare(const CalibCompare&) = delete;
	void operator=(const CalibCompare&) = delete;
//	double lowE_, highE_, lowEta_, highEta_, lowPhi_, highPhi_;
//	unsigned divE_, divEta_, divPhi_;
	bool withOffset_;
	CalibrationTarget target_;
	IO* options_;
	unsigned debug_;

	double mlpOffset_;
	double mlpSlope_;
	PFClusterCalibration clusterCalibration_;
	Erl_mlp erlCalibration_;

};
}

#endif /*CALIBCOMPARE_H_*/
