#ifndef EXERCISES2_H_
#define EXERCISES2_H_

#include "RecoParticleFlow/PFClusterTools/interface/Calibratable.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"
#include <string>
#include <vector>

namespace pftools {
class Exercises2
{
public:
	Exercises2();
	virtual ~Exercises2();
	
	void calibrateCalibratables(const std::string& sourcefile, const std::string& exercisefile);
	
	//makes performance comparisons for the CalibrationResultWrappers in the source file,
	//and compares calibrations for each target supplied
	void doPlots(const std::string& sourcefile, std::vector<CalibrationTarget>& targets);
};
}

#endif /*EXERCISES2_H_*/
