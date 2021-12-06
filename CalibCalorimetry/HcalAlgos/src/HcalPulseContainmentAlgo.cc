#include "CalibCalorimetry/HcalAlgos/src/HcalPulseContainmentAlgo.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include <cmath>
#include <iostream>

// Function generates a lookup map for a passed-in function (via templated object algoObject,
// which must contain method "calcpair" that spits out (x,y) pair from a type float seed.
// Each map y-value is separated from the previous value by a programmable fractional error
// relative to the previous value.
//
HcalPulseContainmentAlgo::HcalPulseContainmentAlgo(int num_samples,
                                                   double fixedphase_ns,
                                                   bool phaseAsInSim,
                                                   const HcalTimeSlew* hcalTimeSlew_delay)
    : fixedphasens_(fixedphase_ns),
      phaseAsInSim_(phaseAsInSim),
      integrator_(&(HcalPulseShapes().hbShape())),
      hcalTimeSlew_delay_(hcalTimeSlew_delay) {
  init(num_samples);
}

HcalPulseContainmentAlgo::HcalPulseContainmentAlgo(const HcalPulseShape* shape,
                                                   int num_samples,
                                                   double fixedphase_ns,
                                                   bool phaseAsInSim,
                                                   const HcalTimeSlew* hcalTimeSlew_delay)
    : fixedphasens_(fixedphase_ns),
      phaseAsInSim_(phaseAsInSim),
      integrator_(shape),
      hcalTimeSlew_delay_(hcalTimeSlew_delay) {
  init(num_samples);
}

void HcalPulseContainmentAlgo::init(int num_samples) {
  const int binsize_ns = 25;

  // First set up controlling parameters for calculating the correction factor:
  // Integration window size...
  //
  integrationwindowns_ = (double)(binsize_ns * num_samples);

  // First find the point at which time bin "1" exceeds time bin "0",
  // and call that point "time 0".
  //
  for (int shift_ns = 0; shift_ns < binsize_ns; shift_ns++) {
    // Digitize by integrating to find all time sample
    // bin values for this shift.
    //
    double tmin = -(double)shift_ns;
    double bin0val = (double)integrator_(tmin, tmin + binsize_ns);
    double bin1val = (double)integrator_(tmin + binsize_ns, tmin + 2 * binsize_ns);

#if 0
    char s[80];
    sprintf (s, "%7.3f %8.5f %8.5f\n", tmin, bin0val, bin1val);
    edm::LogPrint("HcalPulseContainmentAlgo") << s;
#endif

    if (bin1val > bin0val) {
      time0shiftns_ = shift_ns;
      break;
    }
  }
#if 0
  edm::LogPrint("HcalPulseContainmentAlgo") << "time0shiftns_ = " << time0shiftns_;
#endif
}

std::pair<double, double> HcalPulseContainmentAlgo::calcpair(double truefc) {
  double timeslew_ns = hcalTimeSlew_delay_->delay(std::max(0.0, (double)truefc), HcalTimeSlew::Medium);

  double tmin = 0;
  if (phaseAsInSim_) {  // timePhase as in hcalSimParameters, no time0shift
    tmin = fixedphasens_ - timeslew_ns;
  } else {  // Run 2: timePhase opposite to SIM, time0shift
    double shift_ns = fixedphasens_ - time0shiftns_ + timeslew_ns;
    //edm::LogPrint("HcalPulseContainmentAlgo") << "SHIFT " << fixedphasens_ << " " << time0shiftns_ << " " << timeslew_ns;
    tmin = -shift_ns;
  }
  double tmax = tmin + integrationwindowns_;

  //double integral  = shape_.integrate( tmin, tmax );
  double integral = integrator_(tmin, tmax);
  //edm::LogPrint("HcalPulseContainmentAlgo") << "INTEGRAL " << integral << " " << truefc << " " << tmin << " "  << tmax;
  double corfactor = 1.0 / integral;
  double recofc = (double)truefc * integral;

#if 0
  char s[80];
  sprintf (s, "%8.2f %8.4f %8.4f %8.5f %8.5f %8.5f ",
	   truefc, tmin, tmax, integral, corfactor, recofc);
  edm::LogPrint("HcalPulseContainmentAlgo") << s;
#endif

  std::pair<double, double> thepair(recofc, corfactor);
  return thepair;
}
