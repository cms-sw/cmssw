#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include "CalibCalorimetry/HcalAlgos/src/HcalPulseContainmentAlgo.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"

// Function generates a lookup map for a passed-in function (via templated object algoObject,
// which must contain method "calcpair" that spits out (x,y) pair from a type float seed.
// Each map y-value is separated from the previous value by a programmable fractional error
// relative to the previous value.
//
// Currently this function coded for only monotonically increasing or
// decreasing functions...

#include "CalibCalorimetry/HcalAlgos/interface/genlkupmap.h"

///Generate energy correction factors based on a predetermined phase of the hit + time slew
//
HcalPulseContainmentCorrection::HcalPulseContainmentCorrection(int num_samples,
                                                               float fixedphase_ns,
                                                               float max_fracerror)
{
  HcalPulseContainmentAlgo corFalgo(num_samples, (double)fixedphase_ns);

  // Generate lookup map for the correction function, never exceeding
  // a maximum fractional error for lookups.
  //
  //  static const double max_recofc = 5000.0f;   // HPD,  
  //                      max_recofc = 200000.0f; // SiPMs  
  genlkupmap<HcalPulseContainmentAlgo> (1.0, 200000.0f,  // generation domain
                                     max_fracerror,     // maximum fractional error
                                     1.0,   // min_xstep = minimum true fC increment
                                     corFalgo,
                                     mCorFactors_);     // return lookup map
}

// do the same, but with a shape passed in 
HcalPulseContainmentCorrection::HcalPulseContainmentCorrection(
  const HcalPulseShape * shape, 
  int num_samples,
  float fixedphase_ns,
  float max_fracerror)
{
  HcalPulseContainmentAlgo corFalgo(shape, num_samples, (double)fixedphase_ns);
  genlkupmap<HcalPulseContainmentAlgo> (1.0, 200000.0f,  // generation domain
                                     max_fracerror,     // maximum fractional error
                                     1.0,   // min_xstep = minimum true fC increment
                                     corFalgo,
                                     mCorFactors_);     // return lookup map
}

double HcalPulseContainmentCorrection::getCorrection(double fc_ampl) const
{
  double correction;

  std::map<double,double>::const_iterator fcupper,fclower;

  fcupper = mCorFactors_.upper_bound(fc_ampl);
  fclower = fcupper;
  fclower--;

  if (fcupper == mCorFactors_.end()) {
    correction = fclower->second;
  }
  else if (fcupper == mCorFactors_.begin()) {
    correction = fcupper->second;
  }
  else {
    if (fabs(fclower->first - fc_ampl) <
	fabs(fcupper->first - fc_ampl) )
      correction = fclower->second;
    else
      correction = fcupper->second;
  }

#if 0
  char s[80];
  sprintf (s, "%7.1f (%8.5f %8.5f) (%8.5f %8.5f) %8.5f",
	   fc_ampl,
	   fclower->first, fclower->second,
	   fcupper->first, fcupper->second,
	   correction);
  cout << s << endl;
#endif

  return correction;
}
