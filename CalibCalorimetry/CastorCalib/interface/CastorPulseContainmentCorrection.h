#ifndef CALIBCALORIMETRY_CASTORALGOS_CASTORPULSECONTAINMENTCORRECTION_H
#define CALIBCALORIMETRY_CASTORALGOS_CASTORPULSECONTAINMENTCORRECTION_H 1

#include <map>

/** \class CastorPulseContainmentCorrection
  * Amplitude correction for pulse containment in time.
  * Currently only for HPD pulse shape.
  *  
  **/

class CastorPulseContainmentCorrection {
public:
  CastorPulseContainmentCorrection(int num_samples,
                                 float fixedphase_ns,
                                 float max_fracerror);

  double getCorrection(double fc_ampl) const;
  double fractionContained(double fc_ampl) const { return 1.0/this->getCorrection(fc_ampl); }

private:
  std::map<double,double> mCorFactors_;
};

#endif
