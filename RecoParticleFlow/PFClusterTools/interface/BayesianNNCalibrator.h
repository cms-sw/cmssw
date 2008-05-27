#ifndef BAYESIANNNCALIBRATOR_H_
#define BAYESIANNNCALIBRATOR_H_
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"

namespace pftools {
class BayesianNNCalibrator : public Calibrator
{
public:
	BayesianNNCalibrator();
	virtual ~BayesianNNCalibrator();
};

}
#endif /*BAYESIANNNCALIBRATOR_H_*/
