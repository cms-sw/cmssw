#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm> // for "max","min"
#include <math.h>
#include <iostream>

// Function generates a lookup map for a passed-in function (via templated object algoObject,
// which must contain method "calcpair" that spits out (x,y) pair from a type float seed.
// Each map y-value is separated from the previous value by a programmable fractional error
// relative to the previous value.
//
// Currently this function coded for only monotonically increasing or
// decreasing functions...

#include "CalibCalorimetry/HcalAlgos/interface/genlkupmap.h"

template <>
RecoFCcorFactorAlgo<HcalPulseShapes::Shape>::RecoFCcorFactorAlgo(int num_samples, double fixedphase_ns)
: fixedphasens_(fixedphase_ns),
  shape_(HcalPulseShapes().hbShape()),
  integrator_(new HcalShapeIntegrator(&shape_)) 
{
  const int binsize_ns = 25;

  // First set up controlling parameters for calculating the correction factor:
  // Integration window size...
  //
  integrationwindowns_ = (double)(binsize_ns*num_samples);

  // First find the point at which time bin "1" exceeds time bin "0",
  // and call that point "time 0".
  //
  for (int shift_ns=0; shift_ns<binsize_ns; shift_ns++) {

    // Digitize by integrating to find all time sample
    // bin values for this shift.
    //
    double tmin    = -(double)shift_ns;
    double bin0val = (double) (*integrator_)(tmin, tmin+binsize_ns);
    double bin1val = (double) (*integrator_)(tmin+binsize_ns, tmin+2*binsize_ns);

#if 0
    char s[80];
    sprintf (s, "%7.3f %8.5f %8.5f\n", tmin, bin0val, bin1val);
    cout << s;
#endif

    if (bin1val > bin0val) {
      time0shiftns_ = shift_ns;
      break;
    }
  }

#if 0
  cout << "time0shiftns_ = " << time0shiftns_ << endl;
#endif
}

template <>
std::pair<double,double> 
RecoFCcorFactorAlgo<HcalPulseShapes::Shape>::calcpair(double truefc)
{
  double timeslew_ns = HcalTimeSlew::delay(std::max(0.0,(double)truefc),
                                           HcalTimeSlew::Medium);
  double shift_ns  = fixedphasens_ - time0shiftns_ + timeslew_ns;

  double tmin      = -shift_ns;
  double tmax      = tmin+integrationwindowns_;

  //double integral  = shape_.integrate( tmin, tmax );
  double integral = (integrator_) ? (*integrator_)(tmin, tmax) 
                                  : shape_.integrate( tmin, tmax );
  double corfactor = 1.0/integral;
  double recofc    = (double)truefc * integral;

#if 0
  char s[80];
  sprintf (s, "%8.2f %8.4f %8.4f %8.5f %8.5f %8.5f ",
	   truefc, tmin, tmax, integral, corfactor, recofc);
  cout << s;
#endif

  std::pair<double,double> thepair(recofc,corfactor);
  return thepair;
}

///Generate energy correction factors based on a predetermined phase of the hit + time slew
//
HcalPulseContainmentCorrection::HcalPulseContainmentCorrection(int num_samples,
                                                               float fixedphase_ns,
                                                               float max_fracerror)
{
  RecoFCcorFactorAlgo<HcalPulseShapes::Shape> corFalgo(num_samples, (double)fixedphase_ns);

  // Generate lookup map for the correction function, never exceeding
  // a maximum fractional error for lookups.
  //
  //  static const double max_recofc = 5000.0f;
  genlkupmap< RecoFCcorFactorAlgo<HcalPulseShapes::Shape> > (1.0, 5000.0f,  // generation domain
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
