#include "CalibCalorimetry/CastorCalib/interface/CastorPulseContainmentCorrection.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorTimeSlew.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorPulseShapes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm> // for "max","min"
#include <cmath>
#include <iostream>

#include "CalibCalorimetry/HcalAlgos/interface/genlkupmap.h"


template<>
RecoFCcorFactorAlgo<CastorPulseShapes::Shape>::RecoFCcorFactorAlgo(int num_samples, double fixedphase_ns)
{
  fixedphasens_ = fixedphase_ns;
  CastorPulseShapes shapes;
  shape_=shapes.castorShape();
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
    double bin0val = (double)shape_.integrate(tmin, tmin+binsize_ns);
    double bin1val = (double)shape_.integrate(tmin+binsize_ns, tmin+2*binsize_ns);

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
std::pair<double,double> RecoFCcorFactorAlgo<CastorPulseShapes::Shape>::calcpair(double truefc)
{
  double timeslew_ns = CastorTimeSlew::delay(std::max(0.0,(double)truefc),
                                           CastorTimeSlew::Medium);
  double shift_ns  = fixedphasens_ - time0shiftns_ + timeslew_ns;

  double tmin      = -shift_ns;
  double tmax      = tmin+integrationwindowns_;

  double integral  = shape_.integrate( tmin, tmax );
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
CastorPulseContainmentCorrection::CastorPulseContainmentCorrection(int num_samples,
                                                               float fixedphase_ns,
                                                               float max_fracerror)
{
  RecoFCcorFactorAlgo<CastorPulseShapes::Shape> corFalgo(num_samples, (double)fixedphase_ns);

  // Generate lookup map for the correction function, never exceeding
  // a maximum fractional error for lookups.
  //
  genlkupmap< RecoFCcorFactorAlgo<CastorPulseShapes::Shape> > (1.0, 5000.0f,  // generation domain
                                     max_fracerror,     // maximum fractional error
                                     1.0,   // min_xstep = minimum true fC increment
                                     corFalgo,
                                     mCorFactors_);     // return lookup map
}


double CastorPulseContainmentCorrection::getCorrection(double fc_ampl) const
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
