#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm> // for "max","min"

#include <iostream>

// Function generates a lookup map for a passed-in function (via templated object algoObject,
// which must contain method "calcpair" that spits out (x,y) pair from a type float seed.
// Each map y-value is separated from the previous value by a programmable fractional error
// relative to the previous value.
//
// Currently this function coded for only monotonically increasing or
// decreasing functions...

template <class T_algo>
void genlkupmap(double smin,
                double smax,
                double max_fracerror,
                double min_sstep,
                T_algo& algoObject,
                std::map<double,double>& m_ylookup)
{
  std::pair<double,double> thisxy, lastxy, laststoredxy;
  std::pair<double,double> minxy = algoObject.calcpair(smin);
  std::pair<double,double> maxxy = algoObject.calcpair(smax);
  
  double slope = maxxy.second - minxy.second;
  slope = (slope >= 0.0) ? 1.0 : -1.0;

  double sstep = min_sstep;

  for (double s=smin; lastxy.first<smax; s += sstep) {

    thisxy = algoObject.calcpair(s);

    double fracerror  = slope*(thisxy.second - laststoredxy.second)/ thisxy.second;
    double fracchange = slope*(thisxy.second - lastxy.second)/ thisxy.second;

    bool store_cur_pair  = false;
    bool store_prev_pair = false;

#if 0
    char str[80];
    sprintf(str, "%7.1f %7.1f (%8.3f %8.4f) %8.5f %8.5f",
	    s, sstep, thisxy.first, thisxy.second, fracerror, fracchange);
    cout << str;
#endif

    if (s == smin) {
      store_cur_pair = true;
    }
    else if ((fracerror > 2*max_fracerror) ||
             (thisxy.first > smax) ) {
      if (sstep > min_sstep) {
        // possibly overstepped the next entry, back up and reduce the step size
        s -= sstep;
        sstep = std::max(0.5*sstep, min_sstep);
#if 0
	cout << endl;
#endif
        continue;
      }
      else if (lastxy.second == laststoredxy.second) {
        store_cur_pair = true;

        // current step size is too big for the max allowed fractional error,
        // store current value and issue a warning.
        //
	edm::LogWarning("HcalPulseContainmentCorrection::genlkupmap") << " fractional error max exceeded";
      }
      else {
        store_prev_pair = true;

        // re-evaluate current yval with prev yval.
        fracerror = slope*(thisxy.second - lastxy.second)/ thisxy.second;

        if (fracerror > 2*max_fracerror) {
          store_cur_pair = true;

          // current step size is too big for the max allowed fractional error,
          // store current value and issue a warning.
          //
	  edm::LogWarning("HcalPulseContainmentCorrection::genlkupmap") << " fractional error max exceeded";
        }
      }
    }
    else if ((fracerror  < 1.9*max_fracerror) &&
             (fracchange < 0.1*max_fracerror) &&
             (thisxy.first < 0.99 * smax) ) {
      // adapt step size to reduce iterations
      sstep *= 2.0;
    }

    if (thisxy.first > smax)
      store_cur_pair = true;
      
    if (store_prev_pair) {
      m_ylookup[lastxy.first] = lastxy.second;
      laststoredxy            = lastxy;
    }
    if (store_cur_pair) {
      m_ylookup[thisxy.first] = thisxy.second;
      laststoredxy            = thisxy;
    }

    lastxy = thisxy;
    
#if 0
    sprintf(str, " %c %c",
	    store_cur_pair ? 'C' : ' ',
	    store_prev_pair ? 'P' : ' ');
    cout << str << endl;
#endif

  }

}

//======================================================================
// Fixed Phase mode amplitude correction factor generation routines:

class RecoFCcorFactorAlgo {
public:
  RecoFCcorFactorAlgo(int    num_samples,
                      double fixedphase_ns);

  std::pair<double,double> calcpair(double);

private:
  double fixedphasens_;
  double integrationwindowns_;
  double time0shiftns_;
  HcalPulseShapes::Shape shape_;
};

RecoFCcorFactorAlgo::RecoFCcorFactorAlgo(int num_samples, double fixedphase_ns)
{
  fixedphasens_ = fixedphase_ns;
  HcalPulseShapes shapes;
  shape_=shapes.hbShape();
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

static const double max_recofc = 5000.0f;

std::pair<double,double> RecoFCcorFactorAlgo::calcpair(double truefc)
{
  double timeslew_ns = HcalTimeSlew::delay(std::max(0.0,(double)truefc),
                                           HcalTimeSlew::Medium);
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
HcalPulseContainmentCorrection::HcalPulseContainmentCorrection(int num_samples,
                                                               float fixedphase_ns,
                                                               float max_fracerror)
{
  RecoFCcorFactorAlgo corFalgo(num_samples, (double)fixedphase_ns);

  // Generate lookup map for the correction function, never exceeding
  // a maximum fractional error for lookups.
  //
  genlkupmap< RecoFCcorFactorAlgo > (1.0, max_recofc,  // generation domain
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
